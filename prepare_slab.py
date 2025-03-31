from OpenCGChromatin import *
import numpy as np
import math
import sys  # used for flush print statements

Na = 1
Np = 2400 * Na

print("Starting simulation setup...", flush=True)

# -----------------------------
# Helper functions
# -----------------------------

# Function to generate ACTG repeats of arbitrary length
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
# Build the Chromatin Fiber & Protein
# -----------------------------

# Set linker DNA length
linker_length = 30

print("Generating DNA sequence...", flush=True)
# Create DNA sequence for a regularly spaced array of 4 nucleosomes 
DNA_sequence = generate_even_dyads_sequence(12, linker_length)

# Set nucleosome sequence (default core with all tails attached; labelled as '1kx5')
nucleosome_sequence = 12 * ['1kx5']

print("Building dyad positions...", flush=True)
# Build dyad positions along the DNA
first_dyad = linker_length//2 + 73
dyad_positions = [first_dyad]
for i in range(1, 12):
    dyad_positions.append(first_dyad + i * (147 + linker_length))

print("Instantiating NucleosomeArray...", flush=True)
# Instantiate the NucleosomeArray 
nucleosome_array = NucleosomeArray(nucleosome_sequence, DNA_sequence, dyad_positions)

# Get topology and relaxed coordinates (in nm)
na_topology = nucleosome_array.topology

print("Loading relaxed chromatin fiber from PDB...", flush=True)
# Load the relaxed chromatin fiber from a PDB file
pdb_relaxed = app.PDBFile('relaxed_chromatin.pdb')
na_positions = np.array(pdb_relaxed.positions.value_in_unit(unit.nanometer))

# Define array offset positions (single copy at origin for now)
na_offsets = [np.zeros(3)]

print("Creating initial model with NA copy at origin...", flush=True)
# Now create an empty model starting from the relaxed chromatin fiber.
# (We start with one copy of the nucleosome array.)
model = app.Modeller(na_topology, na_positions * unit.nanometer)
# --- Add NA copies using a grid pattern ---
# Assume one NA copy is already at the origin in the model.
na_required = Na - 1  # additional NA copies to add
na_grid_dim = math.ceil(na_required ** (1/3))
na_offsets = [np.array([0, 0, 0])]  # record origin offset

na_added = 0
print("Adding additional NA copies...", flush=True)
for i in range(na_grid_dim):
    for j in range(na_grid_dim):
        for k in range(na_grid_dim):
            # Skip the origin copy if already present.
            if (i, j, k) == (0, 0, 0):
                continue
            if na_added >= na_required:
                break
            offset = 10 * np.array([i, j, k])
            na_offsets.append(offset)
            model.add(na_topology, (na_positions + offset) * unit.nanometer)
            na_added += 1
            print(f"Added NA copy with offset: {offset}", flush=True)
        if na_added >= na_required:
            break
    if na_added >= na_required:
        break
print(f"Total NA copies added (including origin): {na_added + 1}", flush=True)


# --- Instantiate and relax proteins ---
print("Relaxing H1 protein...", flush=True)
fus_seq = 'MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS'
fus = IDP('fus', fus_seq)
fus.relax()  
p_topology = fus.topology
p_positions = fus.relaxed_coords


print("Computing intrinsic centers and radii...", flush=True)
# --- Compute intrinsic centers and effective radii for collision checking ---
def get_center_and_radius(pos_array):
    center = np.mean(pos_array, axis=0)
    radius = np.max(np.linalg.norm(pos_array - center, axis=1))
    return center, radius

# For NA we already computed these earlier
na_center, na_radius = get_center_and_radius(na_positions)
# For H1 and protA (their intrinsic coordinates)
H1_center, H1_radius = get_center_and_radius(p_positions)

# Minimum clearance (in nm) between any two objects (tweak as needed)
min_separation = 1.0

# --- Combined protein placement using pre-calculated safe grid positions ---
print("Preparing grid for protein placement...", flush=True)
desired_H1 = Np

# Define grid parameters for candidate positions (in nm)
candidate_spacing = 15.0  # spacing between candidate grid points

# Calculate grid dimension based on total desired proteins
total_desired = desired_H1 # + desired_protA
grid_dim = math.ceil(total_desired ** (1/3))
print(f"Calculated grid dimension: {grid_dim}", flush=True)

# Generate candidate grid positions
candidate_positions = []
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            pos = candidate_spacing * np.array([i, j, k])
            candidate_positions.append(pos)
candidate_positions = np.array(candidate_positions)
print(f"Generated {candidate_positions.shape[0]} candidate grid positions.", flush=True)

# Filter out grid positions that overlap with NA copies
print("Filtering candidate grid positions for safety...", flush=True)
safe_positions = []
for pos in candidate_positions:
    safe = True
    for na_off in na_offsets:
        na_copy_center = na_off + na_center
        if np.linalg.norm(pos - na_copy_center) < (na_radius + min_separation):
            safe = False
            break
    if safe:
        safe_positions.append(pos)
safe_positions = np.array(safe_positions)
print(f"Found {safe_positions.shape[0]} safe grid positions.", flush=True)

if safe_positions.shape[0] < total_desired:
    raise ValueError("Not enough safe grid positions for the desired number of proteins. Adjust box size or spacing.")

# Shuffle safe positions to randomize placement
np.random.shuffle(safe_positions)

print("Placing H1 proteins...", flush=True)
for i in range(desired_H1):
    candidate_offset = safe_positions[i]
    # Place H1 by translating its relaxed coordinates by the candidate offset
    H1_position = p_positions + candidate_offset
    model.add(p_topology, H1_position * unit.nanometer)
print(f"Placed {desired_H1} H1 copies.", flush=True)

desired_protA = 0  # Initialize desired_protA
print(f"Total: Added {na_added + 1} NA copies (including the one at the origin), {desired_H1} FUS-LCD copies.", flush=True)
print('Total number of particles:', model.topology.getNumAtoms(), flush=True)

app.PDBFile.writeFile(model.topology, model.positions, open('start_model.pdb', 'w'))


# Convert positions to numpy array without OpenMM units for internal functions
model_positions = np.array(model.positions.value_in_unit(unit.nanometer))
print("Converted model positions to numpy array.", flush=True)

"""
pdb = app.PDBFile('start_NVT.pdb')
model_positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
model_topology = pdb.topology

def get_seq(chain):
    return ''.join([residue.name.replace('p', '') for residue in chain.residues()])

for i, chain in enumerate(model_topology.chains()):
    if i < 12:
        chain.id = f'nuc_1kx5_0_{i}'
    elif i == 12:
        chain.id = 'array_DNA_0'
    
    if get_seq(chain) == H1_seq:
        chain.id = H1.chain_id
    elif get_seq(chain) == protA_seq:
        chain.id = protA.chain_id

"""
# -----------------------------
# Set periodic box for initial simulation
# -----------------------------

print("Setting periodic box...", flush=True)
# Define a cubic box with periodic boundary conditions to contain all the particles
box_length = np.max(np.ptp(model_positions, axis=0)) + 2.5  # add small padding
box_vecs = box_length * np.eye(3)
model.topology.setPeriodicBoxVectors(box_vecs)
print("Periodic box set.", flush=True)


# Create the OpenMM System object.
print("Creating OpenMM System object...", flush=True)
debye = 0.8
system = get_system(
        model_positions,
        model.topology,
        {nucleosome_array.chain_id: nucleosome_array.globular_indices, fus.chain_id: fus.globular_indices},
        [nucleosome_array.dyad_positions],
        debye_length=debye,
        constraints='all',
        qAA=1.,
        qP=1.,
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
print("Starting Simulation run 1 (NPT with barostat)...", flush=True)
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
