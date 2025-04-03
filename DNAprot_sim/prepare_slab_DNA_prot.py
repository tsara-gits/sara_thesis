from OpenCGChromatin_Ca import IDP, DNA, get_minimized_system, PLATFORM, PROPERTIES
import numpy as np
import math
import openmm as mm
import openmm.unit as unit
from openmm import app
import os

# ------------------------------------------
# PARAMETERS
# ------------------------------------------

# --------- DNA parameters --------
# w601 DNA sequence
nucl_dna = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'
Ndna = 2                         # number of DNA strands
dna_length = 100                 # number of nucleotides in one strand


# --------- Protein parameters -----------
# Note: the code setup works only for fully IDP proteins

seq_FUS = ("MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS")
chainID_FUS = 'fus'              # OpenMM topology chain ID name (4th coloumn in PDB file)
N_FUS = 70                       # desired number of protein chains to place

# Input here other protein types to if you wish to create a protein mixture 
'''
seq_NPM1 = ("MLQEQSELMSTVMNNTPTTVAALAAVAAASETNGKLGSEEQPEITIPKPRSSAQLEQLLYRYRAIQNHPKENKLEIKAIEDTFRNISRDQDIYETKLDTLRKSIDKGFQYDEDLLNKHLVALQLLEKDTDVPDYFLDLPDTKNDNTTAIEVDYSEKKPIKISADFNAKAKSLGLESKFSNATKTALGDPDTEIRISARISNRINELERLPANLGTYSLDDCLEFITKDDLSSRMDTFKIKALVELKSLKLLTKQKSIRQKLINNVASQAHHNIPYLRDSPFTAAAQRSVQIRSKVIVPQTVRLAEELERQQLLEKRKKERNLHLQKINSIIDFIKERQSEQWSRQERCFQFGRLGASLHNQMEKDEQKRIERTAKQRLAALKSNDEEAYLKLLDQTKDTRITQLLRQTNSFLDSLSEAVRAQQNEAKILHGEEVQPITDEEREKTDYYEVAHRIKEKIDKQPSILVGGTLKEYQLRGLEWMVSLYNNHLNGILADEMGLGKTIQSISLITYLYEVKKDIGPFLVIVPLSTITNWTLEFEKWAPSLNTIIYKGTPNQRHSLQHQIRVGNFDVLLTTYEYIIKDKSLLSKHDWAHMIIDEGHRMKNAQSKLSFTISHYYRTRNRLILTGTPLQNNLPELWALLNFVLPKIFNSAKTFEDWFNTPFANTGTQEKLELTEEETLLIIRRLHKVLRPFLLRRLKKEVEKDLPDKVEKVIKCKLSGLQQQLYQQMLKHNALFVGAGTEGATKGGIKGLNNKIMQLRKICNHPFVFDEVEGVVNPSRGNSDLLFRVAGKFELLDRVLPKFKASGHRVLMFFQMTQVMDIMEDFLRMKDLKYMRLDGSTKTEERTEMLNAFNAPDSDYFCFLLSTRAGGLGLNLQTADTVIIFDTDWNPHQDLQAQDRAHRIGQKNEVRILRLITTDSVEEVILERAMQKLDIDGKVIQAGKFDNKSTAEEQEAFLRRLIESETNRDDDDKAELDDDELNDTLARSADEKILFDKIDKERMNQERADAKAQGLRVPPPRLIQLDELPKVFREDIEEHFKKEDSEPLGRIRQKKRVYYDDGLTEEQFLEAVEDDNMSLEDAIKKRREARERRRLRQNGTKENEIETLENTPEASETSLIENNSFTAAVDEETNADKETTASRSKRRSSRKKRTISIVTAEDKENTQEESTSQENGGAKVEEEVKSSSVEIINGSESKKKKPKLTVKIKLNKTTVLENNDGKRAEEKPESKSPAKKTAAKKTKTKSKSLGIFPTVEKLVEEMREQLDEVDSHPRTSIFEKLPSKRDYPDYFKVIEKPMAIDIILKNCKNGTYKTLEEVRQALQTMFENARFYNEEGSWVYVDADKLNEFTDEWFKEHSS")
chainID_NPM1 = 'npm1'
N_NPM1 = 30
'''

# ----- Lists storing parameters for protein mixtures ------
prot_FUS = IDP(chainID_FUS, seq_FUS)            # initialize IDP biomolecule objects for each protein type
#prot_NPM1 = IDP(chainID_NPM1, seq_NPM1)
proteins_list = [prot_FUS]                      # store the IDP biomolecule objects in a list ([prot_FUS, prot_NPM1, ...])
Np_list = [N_FUS]                               # store the number of proteins for each IDP variant in a list ([N_FUS, N_NPM1, ...])


# ------ Parameters for grid spacing -----------
min_separation = 1.0             # minimum spacing (in nm) between DNA strands - proteins
protein_spacing = 15.0           # grid spacing between protein - protein grid points (in nm)
DNA_spacing = 5.0                # grid spacing between DNA strand - DNA strand grid points (in nm)


# ------ Simluation time parameters -------
sim_time_compress = 200          # simulation time of run 1: NPT with barostat to compress the box (ns)
sim_time_relax = 200             # simulation time of run 2: NVT with extended box for relaxation (ns)




# --------------------------------------
# Helper functions
# --------------------------------------

# Create output directories save the initial pdb structures. Helpful for simulation setup checks.
os.makedirs('OUTPUTS_pdbmodels', exist_ok=True)


def generate_actg_sequence(n):
    '# Generate ACTG repeats of arbitrary length'
    pattern = 'ACTG'
    return (pattern * (n // len(pattern) + 1))[:n]

def generate_even_dyads_sequence(N_nucleosomes, linker_length):
    '''
    Function to generate a combination of ACTG linkers/w601 nucleosomal DNA corresponding to 
    a nucleosome array with evenly spaced nucleosome dyads
    '''
    
    DNA_sequence = generate_actg_sequence(linker_length//2)
    for _ in range(N_nucleosomes-1):
        DNA_sequence += nucl_dna + generate_actg_sequence(linker_length)
    DNA_sequence += nucl_dna + generate_actg_sequence(linker_length//2)
    return DNA_sequence

def calc_DNA_length(dna):
    '''
    Calculates the horizontal and vertical length for the DNA strand 
    '''
    mins = np.min(dna.relaxed_coords, axis=0)
    maxs = np.max(dna.relaxed_coords, axis=0)
    extent = maxs - mins
    
    DNA_length_vertical = extent[2]  # height in nanometers

    DNA_span_x = extent[0]  # x-y span
    DNA_span_y = extent[1]
    
    DNA_length_horizontal = max(DNA_span_x, DNA_span_y)
    return DNA_length_vertical, DNA_length_horizontal

print(f"Starting simulation setup with: \n - {Ndna} DNA strands", flush=True)
for i in range(len(proteins_list)):
    print(f" - {Np_list[i]} {proteins_list[i].chain_id} proteins\n")
print("Helper functions defined.", flush=True)



# -------------------------------------------
# Build DNA and protein mix
# -------------------------------------------

# ---------------- 1. Build and relax DNA strands ----------------
if Ndna != 0:

    print("Initializing similation set up for DNA strands.")
    dna_seq = generate_actg_sequence(dna_length)
    dna = DNA('dna1', dna_seq)
    print(" - Relaxing DNA strand...", flush=True)
    dna.relax()
    print("   DNA strand is relaxed.", flush=True)

    # Save the pdb structure files
    dna_model         = app.Modeller(dna.topology, dna.relaxed_coords * unit.nanometer)  
    dna_model_relaxed = app.Modeller(dna.topology, dna.initial_coords * unit.nanometer)
    app.PDBFile.writeFile(dna_model.topology, dna_model.positions, open(os.path.join('OUTPUTS_pdbmodels', 'dna_model.pdb'), 'w'))
    app.PDBFile.writeFile(dna_model_relaxed.topology, dna_model_relaxed.positions, open(os.path.join('OUTPUTS_pdbmodels', 'dna_relaxed_model.pdb'), 'w'))


# -------------- 2. Place DNA strands in a grid pattern -------------
if Ndna != 0:

    print(" - Placing DNA strands around the origin...", flush=True)
    model = app.Modeller(dna.topology, dna.relaxed_coords * unit.nanometer)   # initialize the modeller and add one strand to the origin
    Ndna_required = Ndna - 1                                                  # one DNA copy is already at the origin in the model
    DNA_length_vertical, DNA_length_horizontal = calc_DNA_length(dna)         # calculate the bounding box of DNA

    dna_grid_dim = math.ceil(Ndna_required ** (1/3))        # calculate the number of grid points needed to place Ndna strands
    Ndna_added = 0                                          # initialize counting of DNA strands added
    dna_offsets = [np.array([0, 0, 0])]                     # initialize the origin offset
    for i in range(dna_grid_dim):
        for j in range(dna_grid_dim):
            for k in range(dna_grid_dim):
                if (i, j, k) == (0, 0, 0):
                    continue                                # skip the origin copy as a DNA strand is already addeed 
                if Ndna_added >= Ndna_required:
                    break
                offset = np.array([i * (DNA_length_horizontal + DNA_spacing), 
                                   j * (DNA_length_horizontal + DNA_spacing), 
                                   k * (DNA_length_vertical   + DNA_spacing)])
                dna_offsets.append(offset)
                model.add(dna.topology, (dna.relaxed_coords + offset) * unit.nanometer)  # add a DNA strand with the offset calculated
                Ndna_added += 1
            if Ndna_added >= Ndna_required:
                break
        if Ndna_added >= Ndna_required:
            break
    print("   DNA strands are placed.", flush=True)
    print(f"DNA model is built. Total number of DNA strands placed: {Ndna_added + 1} \n", flush=True)
    
    # Save the pdb structure file
    app.PDBFile.writeFile(model.topology, model.positions, open(os.path.join('OUTPUTS_pdbmodels', 'dna_strands_model.pdb'), 'w'))



# ----------------- 3. Build and relax proteins --------------------------
print("Initializing similation set up for proteins.")
for prot in proteins_list:

    print(f' - Relaxing protein: {prot.chain_id}...', flush=True)
    prot.relax() 
    print(f'   Protein {prot.chain_id} relaxed.', flush=True)
    
    # Save the pdb structure files
    pmodel = app.Modeller(prot.topology, prot.initial_coords * unit.nanometer)
    pmodel_relaxed = app.Modeller(prot.topology, prot.relaxed_coords * unit.nanometer)
    app.PDBFile.writeFile(pmodel.topology, pmodel.positions, open(os.path.join('OUTPUTS_pdbmodels', prot.chain_id + '_model.pdb'), 'w'))
    app.PDBFile.writeFile(pmodel_relaxed.topology, pmodel_relaxed.positions, open(os.path.join('OUTPUTS_pdbmodels', prot.chain_id + '_relaxed_model.pdb'), 'w'))



# ---------------- 4. Cacluate protein grid positions ----------------
# Note: there will be more or less proteins added to the system then Np, to fill out fully the cubic box

print(" - Generating grid for protein placement...", flush=True)  
grid_dim = math.ceil(sum(Np_list) ** (1/3))                           # calculate the number of grid points per dim needed to place Np proteins
candidate_positions = []   
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            pos = protein_spacing * np.array([i, j, k])
            candidate_positions.append(pos)
candidate_positions = np.array(candidate_positions)
np.random.shuffle(candidate_positions)                # shuffle the positions for random protein placing
print(f"   Generated {candidate_positions.shape[0]} candidate protein grid positions with grid dimension {grid_dim}.", flush=True)



# ---------------- 5. Place proteins in a gird pattern -------------

if Ndna == 0:    
    model = app.Modeller(app.Topology() , []* unit.nanometer)  # initialize and empty modeller
    print(" - Placing proteins ...", flush=True)
    placed = 0
    idx = 0
    for i in range(len(proteins_list)):   # loop over the protein types
        prot = proteins_list[i]
        Np = Np_list[i]
        p_positions = candidate_positions[idx:idx+Np]   # choose the positions for proteins_list[i] type of protein
    
        for offset in p_positions:  # loop over the positions and place the proteins
            if placed == 0:
                model.add(prot.topology, prot.relaxed_coords * unit.nanometer)  # place the first protein at origin
            else:
                nextp_position = prot.relaxed_coords + offset
                model.add(prot.topology, nextp_position * unit.nanometer)
            placed += 1
        idx += Np
    print("   Proteins are placed. ")
    print(f'Protein model is built. Total number of proteins placed: {placed}')

if Ndna != 0:    
    dna_coords = np.array(model.positions.in_units_of(unit.nanometer))      # extract the DNA coordinates from the defined coords
    print(" - Placing proteins around DNA strands with overlap check...", flush=True)
    placed = 0
    idx = 0
    for i in range(len(proteins_list)):   # loop over the protein types
        prot = proteins_list[i]
        Np = Np_list[i]
        p_positions = candidate_positions[idx:idx+Np]   # choose the positions for proteins_list[i] type of protein
    
        for offset in p_positions:  # loop over the positions and place the proteins
            nextp_position = prot.relaxed_coords + offset
            
            # Only place protein if all atoms are far enough from DNA
            dists = np.linalg.norm(dna_coords[:, np.newaxis, :] - nextp_position[np.newaxis, :, :], axis=2)
            min_dist = np.min(dists)
            if min_dist >= min_separation:
                model.add(prot.topology, nextp_position * unit.nanometer)
                placed += 1
        idx += Np
    print("   Proteins are placed. ")
    print(f'Protein model is built. Total number of proteins placed: {placed}\n')

app.PDBFile.writeFile(model.topology, model.positions, open(os.path.join('OUTPUTS_pdbmodels', 'start_model.pdb'), 'w'))
print(f'The initial model is set up. \nTotal number of particles: { model.topology.getNumAtoms()}\n', flush=True)



# ----------------------------------------------
# Set periodic box for initial simulation
# ----------------------------------------------
print("Setting periodic box and initializing the simulation.")

# ------- Convert positions to numpy array without OpenMM units for internal functions -----
model_positions = np.array(model.positions.value_in_unit(unit.nanometer))

 # ------ Define a cubic box with periodic boundary conditions to contain all the particles  ----
print(" - Setting periodic box...", flush=True) 
box_length = np.max(np.ptp(model_positions, axis=0)) + 2.5  # add small padding
box_vecs = box_length * np.eye(3)
model.topology.setPeriodicBoxVectors(box_vecs)
print("  Periodic box is set.", flush=True)

# ----------- Set up system  -------------
print(" - Initializing simulation system...", flush=True)
# Since there is no chromatin, create an empty dictionary for globular indices
indeces_dict = {chain.id: [] for chain in model.topology.chains()}
dyad_positions=[]   # no dyad positions
debye = 0.8

# Function to create a system with a given Debye length
def system_wrapper(debye, indeces_dict, dyad_positions):
    return get_minimized_system(
        model_positions,
        model.topology,
        indeces_dict,
        dyad_positions,
        debye_length=debye,
        constraints='breathing',  # no nucleosome/DNA constraints , none ?
        qTail=1.0,
        qCore=1.0,
        qP=1.0,
        overall_LJ_eps_scaling=0.75,
        coul_rc= 2. + 2.0 * debye,
        mechanics_k_coeff=1.00,
        anchor_scaling=0.,
        kappa_factor=0.,
        periodic=True,
        CoMM_remover=True
    )[0]

system = system_wrapper(debye, indeces_dict, dyad_positions)
print(" - Simulation system is initialized.\n", flush=True)



# -----------------------------------
# Simulation run 1: NPT with barostat to compress the box over 200 ns
# -----------------------------------
print("Starting simulation run 1: NPT with barostat to compress the box.", flush=True)

# Add a barostat with external pressure of 0.5 atm
barostat = mm.MonteCarloBarostat(0.5 * unit.atmosphere, 300 * unit.kelvin, 25)
system.addForce(barostat)

# Create the integrator and simulation
integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 0.01 / unit.picosecond, 10 * unit.femtosecond)
simulation = app.Simulation(model.topology, system, integrator, platform=PLATFORM, platformProperties=PROPERTIES)

# Set positions and box vectors, then minimize energy
simulation.context.setPositions(model.positions)
simulation.context.setPeriodicBoxVectors(*model.topology.getPeriodicBoxVectors())
print(" - Minimizing energy...", flush=True)
simulation.minimizeEnergy()
print("  Energy is minimized.", flush=True)

# Set up reporters (adjust intervals as needed)
simulation.reporters.append(app.XTCReporter('traj_barostat.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_barostat.out', reportInterval=10000, step=True,
                                                   potentialEnergy=True, temperature=True, elapsedTime=True))
print(" - Running NPT simulation for 200 ns...", flush=True)
Nsteps_relax = int(sim_time_compress * unit.nanosecond / (10 * unit.femtosecond))
simulation.step(Nsteps_relax)
print("   Simulation run 1 completed.", flush=True)
state = simulation.context.getState(getPositions=True, getVelocities=True)



# -----------------------------
# Remove the barostat and extend the box in x-direction
# -----------------------------
print(" - Removing barostat and extending box in x-direction...", flush=True)
# Remove the MonteCarloBarostat force from the system
for idx in range(system.getNumForces()):
    force = system.getForce(idx)
    if isinstance(force, mm.MonteCarloBarostat):
        system.removeForce(idx)
        print("   Barostat removed.", flush=True)
        break
simulation.context.reinitialize() # ensure the context is reinitialized after removing the barostat
simulation.context.setState(state)

# Get current box vectors, then modify the x-component by a factor of 6
current_box = state.getPeriodicBoxVectors()
new_box = [6 * current_box[0], current_box[1], current_box[2]]
simulation.context.setPeriodicBoxVectors(*new_box)
print(" - Box extended.\n", flush=True)




# -----------------------------
# Simulation run 2: NVT with extended box for 200 ns relaxation
# -----------------------------

print("Starting simulation run 2: NVT with extended box.", flush=True)
# Optionally, update reporters for the second run
simulation.reporters = []  # clear previous reporters if desired
simulation.reporters.append(app.XTCReporter('traj_NVT.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_NVT.out', reportInterval=10000, step=True,
                                                   potentialEnergy=True, temperature=True, elapsedTime=True))
print(" - Running NVT simulation for 200 ns...", flush=True)
Nsteps_relax = int(sim_time_relax * unit.nanosecond / (10 * unit.femtosecond))
simulation.step(Nsteps_relax)
print("   Simulation run 2 complete.", flush=True)

# Save final coordinates to a PDB file
print(" - Saving final coordinates to PDB...", flush=True)
with open('final_model.pdb', 'w') as f:
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
print("   Final model saved.\nSimulation is finished", flush=True)



