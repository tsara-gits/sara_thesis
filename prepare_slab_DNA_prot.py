# File: prepare_slab_DNA.py

from OpenCGChromatin import DNA, get_system, PLATFORM, PROPERTIES
import numpy as np
import math
from openmm import unit, mm
from openmm import app

# --------------- PARAMETERS -------------------
seq = ("MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS")
chain_id = 'fus' # OpenMM topology chain name (4th coloumn in PDB file)
min_separation = 1.0  # Minimum clearance (in nm) between any two objects (tweak as needed)
Np = 100  # number of proteins
dna_length = 100  # number of nucleotides in one strand
total_prot = Np  # number of desired proteins to place, add other proteins if it is a protein mix
candidate_spacing = 15.0  # grid parameters for candidate positions (in nm) | HOW MUCH FOR DNA?

print("Starting simulation setup...", flush=True)


# -----------------------------
# Helper functions
# -----------------------------

# Generate ACTG repeats of arbitrary length
def generate_actg_sequence(n):
    pattern = 'ACTG'
    return (pattern * (n // len(pattern) + 1))[:n]

# Function to generate a combination of ACTG linkers/w601 nucleosomal DNA corresponding to 
# a nucleosome array with evenly spaced nucleosome dyads
def generate_even_dyads_sequence(N_nucleosomes, linker_length):
    w601 = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'
    DNA_sequence = generate_actg_sequence(linker_length//2)
    for _ in range(N_nucleosomes-1):
        DNA_sequence += w601 + generate_actg_sequence(linker_length)
    DNA_sequence += w601 + generate_actg_sequence(linker_length//2)
    return DNA_sequence

print("Helper functions defined.", flush=True)



# -----------------------------
# Build DNA and protein mix
# -----------------------------
print("Starting DNA-only simulation setup...", flush=True)

dna_seq = generate_actg_sequence(dna_length)
dna = DNA('dna1', dna_seq)
dna_topology = dna.topology
dna_positions = dna.generate_initial_coords()
print("Creating initial model with DNA at origin...", flush=True)
model = app.Modeller(dna_topology, dna_positions * unit.nanometer)
print("DNA model is created.", flush=True)

# --- Applying relaxation to DNA strand? ----

# --- Instantiate and relax proteins ---
print("Relaxing H1 protein...", flush=True)
fus_seq = 'MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS'
fus = IDP('fus', fus_seq)
fus.relax()  
p_topology = fus.topology
p_positions = fus.relaxed_coords

# ----------- Compute intrinsic centers and effective radii for collision checking -----------
print("Computing intrinsic centers and radii...", flush=True)
def get_center_and_radius(pos_array):
    center = np.mean(pos_array, axis=0)
    radius = np.max(np.linalg.norm(pos_array - center, axis=1))
    return center, radius
prot_center, prot_radius = get_center_and_radius(p_positions)
dna_center, dna_radius = get_center_and_radius(dna_positions)


# ---------------- Cacluate protein grid positions ----------------
print("Generating grid for protein placement...", flush=True)
grid_dim = math.ceil(total_prot ** (1/3))
print(f"Calculated grid dimension: {grid_dim}", flush=True)

candidate_positions = []   # Generate candidate grid positions
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            pos = candidate_spacing * np.array([i, j, k])
            candidate_positions.append(pos)
candidate_positions = np.array(candidate_positions)
print(f"Generated {candidate_positions.shape[0]} candidate grid positions.", flush=True)


# -------- Filter out grid positions that overlap with the DNA ------------
print("Filtering candidate grid positions for safety...", flush=True) 
safe_positions = []
for pos in candidate_positions:
    safe = True
    if np.linalg.norm(pos - dna_center) < (dna_radius + min_separation):
        safe = False
        break
    if safe:
        safe_positions.append(pos)
safe_positions = np.array(safe_positions)
print(f"Found {safe_positions.shape[0]} safe grid positions.", flush=True)

if safe_positions.shape[0] < total_desired:
    raise ValueError("Not enough safe grid positions for the desired number of proteins. Adjust box size or spacing.")

np.random.shuffle(safe_positions)  # Shuffle safe positions to randomize placement



# ------- Place proteins around the DNA strand --------
print("Placing proteins...", flush=True)
for i in range(Np):
    candidate_offset = safe_positions[i]
    new_position = p_positions + candidate_offset          # Place proteins by translating its relaxed coordinates by the candidate offset
    model.add(p_topology, new_position * unit.nanometer)
print(f"Placed {Np} protein copies.", flush=True)
print(f"Total: Added one DNA strand and {Np} protein copies.", flush=True)
print('Total number of particles:', model.topology.getNumAtoms(), flush=True)
app.PDBFile.writeFile(model.topology, model.positions, open('start_model.pdb', 'w'))

# -----------------------------
# Set periodic box for initial simulation
# -----------------------------

# ---- Convert positions to numpy array without OpenMM units for internal functions ----
model_positions = np.array(model.positions.value_in_unit(unit.nanometer))
print("Converted model positions to numpy array.", flush=True)

 # ------ Define a cubic box with periodic boundary conditions to contain all the particles  ----
print("Setting periodic box...", flush=True) 
box_length = np.max(np.ptp(model_positions, axis=0)) + 2.5  # add small padding
box_vecs = box_length * np.eye(3)
model.topology.setPeriodicBoxVectors(box_vecs)
print("Periodic box set.", flush=True)

# ----------- Set up system  -------------
globular_indices_dict = {}  # empty dictionary since DNA object does not use globular indices ?
dyad_positions = []         # no dyad positions
debye = 0.8

system = get_system(
    model_positions,
    model.topology,
    globular_indices_dict,
    dyad_positions,    # no dyad positions
    debye_length = debye,
    constrains = 'all',
    qAA=1.,
    qP=1.,
    constraints='none',  # no nucleosome/DNA constraints
    overall_LJ_eps_scaling=1.,
    protein_DNA_LJ_eps_scaling=1.,
    coul_rc= 2. + 2.0 * debye,
    mechanics_k_coeff=1.0,
    anchor_scaling=0.,
    periodic=True,
    CoMM_remover=True
)
print("System object created.", flush=True)

# -----------------------------
# Simulation run 1: NPT with barostat to compress the box over 200 ns
# -----------------------------
# Add a barostat with external pressure of 0.5 atm
barostat = mm.MonteCarloBarostat(0.5 * unit.atmosphere, 300 * unit.kelvin, 25)
system.addForce(barostat)

# Create the integrator and simulation
integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 0.01 / unit.picosecond, 10 * unit.femtosecond)
simulation = app.Simulation(model.topology, system, integrator, platform=PLATFORM, platformProperties=PROPERTIES)

# Set positions and box vectors, then minimize energy
simulation.context.setPositions(model.positions)
simulation.context.setPeriodicBoxVectors(*model.topology.getPeriodicBoxVectors())
print("Minimizing energy...", flush=True)
simulation.minimizeEnergy()
steps_200ns = int(200 * unit.nanosecond / (10 * unit.femtosecond))

# Set up reporters (adjust intervals as needed)
simulation.reporters.append(app.XTCReporter('traj_barostat.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_barostat.out', reportInterval=10000, step=True,
                                                   potentialEnergy=True, temperature=True, elapsedTime=True))
print("Running NPT simulation for 200 ns...", flush=True)
steps_200ns = int(200 * unit.nanosecond / (10 * unit.femtosecond))
simulation.step(steps_200ns)
print("Simulation run 1 complete.", flush=True)

state = simulation.context.getState(getPositions=True, getVelocities=True)


# -----------------------------
# Remove the barostat and extend the box in x-direction
# -----------------------------
print("Removing barostat and extending box in x-direction...", flush=True)
# Remove the MonteCarloBarostat force from the system
for idx in range(system.getNumForces()):
    force = system.getForce(idx)
    if isinstance(force, mm.MonteCarloBarostat):
        system.removeForce(idx)
        print("Barostat removed.", flush=True)
        break
simulation.context.reinitialize() # ensure the context is reinitialized after removing the barostat
simulation.context.setState(state)


# Get current box vectors, then modify the x-component by a factor of 6
current_box = state.getPeriodicBoxVectors()
new_box = [6 * current_box[0], current_box[1], current_box[2]]
simulation.context.setPeriodicBoxVectors(*new_box)
print("Box extended.", flush=True)

# -----------------------------
# Simulation run 2: NVT with extended box for 200 ns relaxation
# -----------------------------
print("Starting Simulation run 2 (NVT relaxation)...", flush=True)
# Optionally, update reporters for the second run
simulation.reporters = []  # clear previous reporters if desired
simulation.reporters.append(app.XTCReporter('traj_NVT.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_NVT.out', reportInterval=10000, step=True,
                                                   potentialEnergy=True, temperature=True, elapsedTime=True))
print("Running NVT simulation for 200 ns...", flush=True)
simulation.step(steps_200ns)
print("Simulation run 2 complete.", flush=True)

# Save final coordinates to a PDB file
print("Saving final coordinates to PDB...", flush=True)
with open('final_model.pdb', 'w') as f:
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
print("Final model saved.", flush=True)



