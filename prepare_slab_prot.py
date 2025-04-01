from OpenCGChromatin import IDP, get_system, PLATFORM, PROPERTIES
import numpy as np
import math
from openmm import unit, mm
from openmm import app

# --------------- PARAMETERS -------------------
# FUS protein
seq = ("MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS")
chain_id = 'fus' # OpenMM topology chain name (4th coloumn in PDB file)
min_separation = 1.0  # Minimum clearance (in nm) between any two objects (tweak as needed)
Np = 100  # number of proteins
desired_prot = Np  # add other proteins if it is a protein mix
grid_spacing = 15.0  # Define grid parameters for candidate positions (in nm)


# ----------- Instantiate and relax proteins --------------------
print("Starting protein-only simulation setup...", flush=True)
print("Relaxing protein...", flush=True)
protein = IDP(chain_id, seq)
protein.relax()
p_topology = protein.topology
p_positions = protein.relaxed_coords
print("Protein relaxation complete.", flush=True)


# ----------- Compute intrinsic centers and effective radii for collision checking -----------
print("Computing intrinsic centers and radii...", flush=True)
def get_center_and_radius(pos_array):
    center = np.mean(pos_array, axis=0)
    radius = np.max(np.linalg.norm(pos_array - center, axis=1))
    return center, radius
prot_center, prot_radius = get_center_and_radius(p_positions)


# ------------ Cacluate grid positions -------------

print("Preparing grid for protein placement...", flush=True)
grid_dim = math.ceil(Np ** (1/3))  # Calculate grid dimension based on total desired proteins
print(f"Calculated grid dimension: {grid_dim}", flush=True)

# Generate candidate grid positions
candidate_positions = []
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            pos = grid_spacing * np.array([i, j, k])
            candidate_positions.append(pos)
candidate_positions = np.array(candidate_positions)
print(f"Generated {candidate_positions.shape[0]} candidate grid positions.", flush=True)

# Select the first num_proteins positions 
selected_positions = candidate_positions[:Np]
safe_positions = np.random.shuffle(selected_positions)  # Shuffle positions to randomize placement


# ------------------ Place proteins ----------------------

print("Placing protein copies...", flush=True)
model = app.Modeller(p_topology, p_positions * unit.nanometer)  # Start a modeller with the first protein copy

for i in range(desired_prot):
    candidate_offset = safe_positions[i]
    new_position = p_positions + candidate_offset         # Place protein by translating its relaxed coordinates by the candidate offset
    model.add(p_topology, new_position * unit.nanometer)
print(f"Placed {desired_H1} H1 copies.", flush=True)

print(f"Total: Added {desired_prot} protein copies.", flush=True)
print('Total number of particles:', model.topology.getNumAtoms(), flush=True)
app.PDBFile.writeFile(model.topology, model.positions, open('start_model.pdb', 'w'))


# -----------------------------
# Set periodic box for initial simulation
# -----------------------------

# convert positions to numpy array without OpenMM units for internal functions
model_positions = np.array(model.positions.value_in_unit(unit.nanometer))
print("Converted model positions to numpy array.", flush=True)

 # Define a cubic box with periodic boundary conditions to contain all the particles
print("Setting periodic box...", flush=True) 
box_length = np.max(np.ptp(model_positions, axis=0)) + 2.5  # add small padding
box_vecs = box_length * np.eye(3)
model.topology.setPeriodicBoxVectors(box_vecs)
print("Periodic box set.", flush=True)

# For protein-only simulation, create an empty dictionary for globular indices
globular_indices_dict = {chain.id: [] for chain in model.topology.chains()}
debye = 0.8

system = get_system(
    model_positions,
    model.topology,
    globular_indices_dict,
    dyad_positions=[],    # no dyad positions
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


