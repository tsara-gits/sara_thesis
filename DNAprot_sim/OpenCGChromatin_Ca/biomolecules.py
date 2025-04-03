import numpy as np
import openmm.app as app
import openmm as mm
import openmm.unit as unit
import warnings

from .coordinate_building import generate_spiral_coords, generate_array_positions, generate_linear_DNA_coords, parse_pdb
from .system_building import seqtrans, get_system
from .constants import NUCLEOSOME_DATA, PLATFORM, PROPERTIES, DATA_DIR

class CGBiomolecule:
    """
    Base class representing a coarse-grained biomolecule. 

    This class provides the basic structure for a biomolecule, including methods
    for sequence validation, topology creation, and system setup in OpenMM. 
    Subclasses must implement methods for generating specific topologies and initial coordinates.

    Attributes:
        chain_id (str): Identifier for the biomolecular chain.
        sequence (str): Sequence of the biomolecule (protein or DNA).
        globular_indices (list): List of residue indices that form the globular (structured) part of the molecule.
        dyad_positions (list, optional): Positions of dyads in the molecule (used for nucleosome/DNA structures).
        topology (Topology): OpenMM Topology object for the biomolecule.
        initial_coords (ndarray): Initial 3D coordinates for the biomolecule.
    """

    def __init__(self, chain_id, sequence, valid_residues='', globular_indices=[], dyad_positions=None):
        """
        Initializes a CGBiomolecule object with the given chain ID, sequence, and optional attributes.

        Args:
            chain_id (str): Identifier for the biomolecular chain.
            sequence (str): Sequence of the biomolecule (amino acids or nucleotides).
            valid_residues (str, optional): A string containing valid residues. Defaults to an empty string.
            globular_indices (list, optional): Indices of residues that form the globular region. Defaults to an empty list.
            dyad_positions (list, optional): Positions of dyads in the molecule (used for DNA structures). Defaults to None.
        """
        self.chain_id = chain_id
        self.sequence = sequence
        self.globular_indices = globular_indices
        self.validate_sequence(set(valid_residues))  # Validate the sequence with provided valid residues
        #self.topology = self.create_monomer_topology()
        self.initial_coords = self.generate_initial_coords()
        self.dyad_positions = dyad_positions

    def validate_sequence(self, valid_entries):
        """
        Validates the biomolecule's sequence to ensure it contains only valid residue entries.

        Args:
            valid_entries (set): A set of valid residue or nucleotide symbols.

        Raises:
            ValueError: If the sequence contains symbols not present in the set of valid entries.
        """
        invalid_entries = set(self.sequence) - valid_entries
        if invalid_entries:
            raise ValueError(f"Invalid sequence. Found invalid entries: {invalid_entries}. Allowed: {valid_entries}")


    def create_monomer_topology(self):
        """
        Creates the topology for the biomolecule. 
        
        This is an abstract method that must be implemented by subclasses to define the
        specific topology for the biomolecule type.

        Returns:
            Topology: An OpenMM Topology object for the biomolecule.
        """
        raise NotImplementedError("Subclasses must implement create_monomer_topology method")

    def generate_initial_coords(self):
        """
        Generates the initial coordinates for the biomolecule.

        This is an abstract method that must be implemented by subclasses to define
        the specific coordinate generation strategy for the biomolecule type.

        Returns:
            ndarray: Initial 3D coordinates for the biomolecule.
        """
        raise NotImplementedError("Subclasses must implement generate_initial_coords method")

    def relax(
        self,
        steps=100000,
        temperature=300*unit.kelvin,
        friction=0.1/unit.picosecond,
        dt=2*unit.femtosecond,
        periodic=False,
        constraints='breathing',
        CoMM_remover=True,
        platform=PLATFORM,
        properties=PROPERTIES
    ):
        """
        Relax the biomolecule by:
          1) Minimizing energy
          2) Running short dynamics (default 100,000 steps)

        The final coordinates are stored in `self.relaxed_coords`.

        Args:
            steps (int): Number of MD steps to run for relaxation.
            temperature (Quantity): Simulation temperature.
            friction (Quantity): Friction coefficient for Langevin integrator.
            dt (Quantity): Timestep for the integrator.
            periodic (bool): Whether or not to use periodic boundary conditions.
            constraints (str): Constraint settings (e.g. 'breathing').
            CoMM_remover (bool): Whether or not to remove center-of-mass motion.
            platform (openmm.Platform): OpenMM platform to use for the simulation.
            properties (dict): Properties for the platform.

        Returns:
            relaxed_coords (ndarray): Relaxed 3D coordinates of the biomolecule.
        """
        # If chain_id is None, we cannot create a single {chain_id: globular_indices} entry.
        # For simpler single-chain classes, we can do so. For multi-chain classes, override this method.
        if self.chain_id is not None:
            chain_indices_dict = {self.chain_id: self.globular_indices}
        else:
            chain_indices_dict = {}  # For multi-chain systems, override if needed

        # For DNA or other classes that don't use dyad_positions, this will be empty
        if self.dyad_positions is not None:
            dyad_list = [self.dyad_positions]
        else:
            dyad_list = []

        system = get_system(
            self.initial_coords,
            self.topology,
            chain_indices_dict,
            dyad_list,
            periodic=periodic,
            constraints=constraints,
            CoMM_remover=CoMM_remover
        )

        integrator = mm.LangevinMiddleIntegrator(temperature, friction, dt)
        simulation = app.Simulation(self.topology, system, integrator, platform, properties)

        # Set initial coordinates
        simulation.context.setPositions(self.initial_coords)
        simulation.minimizeEnergy()

        # Run short dynamics
        simulation.step(steps)

        # Extract final positions
        self.relaxed_coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return self.relaxed_coords


class IDP(CGBiomolecule):
    """
    Class representing an intrinsically disordered protein (IDP).

    Inherits from CGBiomolecule and implements methods specific to IDPs,
    including topology creation and coordinate generation. IDPs are
    modelled as fully-flexible polymers.
    """

    def __init__(self, chain_id, sequence):
        """
        Initializes an IDP object with the given chain ID and sequence.

        Args:
            chain_id (str): Identifier for the protein chain.
            sequence (str): Amino acid sequence of the intrinsically disordered protein.
        """
        super().__init__(chain_id, sequence, valid_residues='ACDEFGHIKLMNPQRSTVWY')
        self.topology = self.create_monomer_topology()

    def create_monomer_topology(self):
        """
        Creates the topology for the IDP using the provided sequence and chain ID.

        Returns:
            Topology: The OpenMM Topology object for the IDP.
        """
        return create_monomer_topology(self.sequence, self.chain_id, chain_type='prt')

    def generate_initial_coords(self):
        """
        Generates initial coordinates for the IDP, modeled as a spiral structure.

        The coordinates are spaced using a default spacing value of 0.381 nm
        between each residue.

        Returns:
            ndarray: Initial 3D coordinates for the IDP in a spiral configuration.
        """
        return generate_spiral_coords(len(self.sequence), spacing=0.381)
       
class MDP(CGBiomolecule):
    """
    Class representing a multi-domain (globular) protein (MDP). Globular domains are constained using an elastic-network model,
    whilst intrinsically-disordered regions are modelled as fully-flexible polymers.

    Inherits from CGBiomolecule and implements methods specific to globular proteins,
    including topology creation and coordinate generation from a PDB file.
    """

    def __init__(self, chain_id, sequence, globular_indices, pdb_file):
        """
        Initializes an MDP object with the given chain ID, sequence, globular indices, and PDB file.

        Args:
            chain_id (str): Identifier for the protein chain.
            sequence (str): Amino acid sequence of the multi-domain protein.
            globular_indices (list): List of residue indices corresponding to globular (structured) regions.
            pdb_file (str): Path to the PDB file containing the initial structure of the protein.
        """
        self.pdb_file = pdb_file
        super().__init__(chain_id, sequence, globular_indices=globular_indices, valid_residues='ACDEFGHIKLMNPQRSTVWY')
        self.topology = self.create_monomer_topology()

    def create_monomer_topology(self):
        """
        Creates the topology for the multi-domain protein using the provided sequence, chain ID, and globular indices.

        Returns:
            Topology: The OpenMM Topology object for the multi-domain protein.
        """
        return create_monomer_topology(self.sequence, self.chain_id, chain_type='prt', globular_indices=self.globular_indices)

    def generate_initial_coords(self):
        """
        Parses the PDB file to generate initial coordinates for the multi-domain protein.
        
        If the sequence in the PDB file does not match the provided sequence, a warning is issued.
        
        Returns:
            ndarray: Initial 3D coordinates for the multi-domain protein from the PDB file.
        """
        coords, pdb_sequence = parse_pdb(self.pdb_file, self.globular_indices)
        if self.sequence != pdb_sequence:
            warnings.warn("Mismatch between provided and PDB sequences.")
        
        return coords

class DNA(CGBiomolecule):
    """
    Class representing a double-stranded DNA molecule, modelled using a forcefield adapted from CGeNArate: https://doi.org/10.1093/nar/gkae444.

    Inherits from CGBiomolecule and implements methods specific to DNA,
    including topology creation and coordinate generation. DNA contains the nucleotide sequence A, C, G, and T.
    The full sequence includes the complementary strand.
    """

    def __init__(self, chain_id, sequence):
        """
        Initializes a DNA object with the given chain ID and sequence.

        The full sequence is generated by concatenating the provided sequence with its complementary strand.

        Args:
            chain_id (str): Identifier for the DNA chain.
            sequence (str): Nucleotide sequence of one strand of the DNA molecule (must contain only A, C, G, and T).
        """
        self.full_sequence = sequence + seqtrans(sequence)
        super().__init__(chain_id, sequence, valid_residues='ACGT')
        self.topology = self.create_monomer_topology()

    def create_monomer_topology(self):
        """
        Creates the topology for the DNA using the full sequence (both strands) and the chain ID.

        Returns:
            Topology: The OpenMM Topology object for the DNA molecule.
        """
        return create_monomer_topology(self.full_sequence, self.chain_id, chain_type='DNA')

    def generate_initial_coords(self):
        """
        Generates initial coordinates for the DNA, modeled as a linear structure.

        Returns:
            ndarray: Initial 3D coordinates for the DNA in a linear configuration.
        """
        return generate_linear_DNA_coords(len(self.sequence))

class NucleosomeCore(CGBiomolecule):
    """
    Class representing a nucleosome core particle, where the 8 separate chains forming the octamer are treated as a 
    single OpenMM Chain object in the model.

    Inherits from CGBiomolecule and implements methods specific to nucleosome core particles. The class reads data from
    the provided NUCLEOSOME_DATA and generates the initial coordinates and topology.
    """

    def __init__(self, nucleosome_id):
        """
        Initializes a NucleosomeCore object using the provided nucleosome ID.

        The class retrieves nucleosome data from NUCLEOSOME_DATA and extracts
        information like sequence, globular indices, and the path to the PDB file.
        Initial coordinates are read from the PDB file.

        Args:
            nucleosome_id (str): Identifier for the nucleosome, used to retrieve data from NUCLEOSOME_DATA.
        """
        self.nucleosome_data = NUCLEOSOME_DATA[nucleosome_id]
        self.chain_id = 'nuc_' + nucleosome_id

        self.N = self.nucleosome_data['N']
        self.IDRs = self.nucleosome_data['IDRs']
        self.pdb_path = DATA_DIR + self.nucleosome_data['aa_pdb_path']
        
        self.globular_indices = [[i for i in range(self.N) if not any(i in IDR for IDR in self.IDRs)]]
        
        self.initial_coords, self.sequence = parse_pdb(self.pdb_path, self.globular_indices)

        super().__init__(self.chain_id, self.sequence, globular_indices=self.globular_indices, valid_residues='ACDEFGHIKLMNPQRSTVWY')
        self.topology = self.create_monomer_topology()

    def create_monomer_topology(self, topology=None):
        """
        Creates the topology for the nucleosome using the sequence, chain ID, and globular indices.

        Args:
            topology (Topology, optional): An existing OpenMM Topology object to which the nucleosome's chain will be added.
                If not provided, a new Topology object will be created.

        Returns:
            Topology: The OpenMM Topology object for the nucleosome.
        """
        return create_monomer_topology(self.sequence, self.chain_id, chain_type='prt', globular_indices=self.globular_indices, topology=topology)

    def generate_initial_coords(self):
        """
        Retrieves the initial coordinates for the nucleosome, which were read from the PDB file during initialization.

        Returns:
            ndarray: Initial 3D coordinates for the nucleosome.
        """
        return self.initial_coords

class NucleosomeArray(CGBiomolecule):
    """
    Class representing an array of nucleosomes with associated DNA.

    Inherits from CGBiomolecule and implements methods specific to nucleosome arrays.
    The class generates the combined topology and coordinates for the nucleosome array,
    using nucleosome and DNA sequences provided at initialization.
    """

    _array_counter = 0
    def __init__(self, nucleosome_id_sequence, DNA_sequence, dyad_positions, from_CG=True):
        """
        Initializes a NucleosomeArray object with the provided nucleosome ID sequence,
        DNA sequence, and dyad positions.

        Args:
            nucleosome_id_sequence (list): List of nucleosome IDs corresponding to the nucleosomes in the array.
            DNA_sequence (str): Nucleotide sequence of the DNA associated with the array.
            dyad_positions (list): List containing the positions along the DNA sequence of nucleosome dyads.
            from_CG (bool, optional): If True, use coarse-grained nucleosome structures for position generation. Defaults to True.
        """
        self.nucleosome_id_sequence = nucleosome_id_sequence
        self.DNA_sequence = DNA_sequence
        self.sequence = self.DNA_sequence + seqtrans(self.DNA_sequence) 
        self.dyad_positions = dyad_positions
        self.from_CG = from_CG
       
        super().__init__(None, self.sequence, valid_residues='ACGT', dyad_positions=dyad_positions)
        self.relaxed_coords = None
        self.topology = self.create_monomer_topology()
       
        
    def create_monomer_topology(self):
        """
        Creates the topology for the nucleosome array, including both the nucleosomes and the DNA.

        The topology is built by creating monomer topologies for each nucleosome in the array,
        and then adding the DNA chain to the array.

        Returns:
            Topology: The OpenMM Topology object representing the entire nucleosome array and associated DNA.
        """
        array_topology = app.Topology()

        array_nuc_count = 0
        for nuc_id in self.nucleosome_id_sequence:
            nuc = NucleosomeCore(nuc_id) 
            nuc.chain_id += f'_{NucleosomeArray._array_counter}_{array_nuc_count}'
            nuc.create_monomer_topology(topology=array_topology)
            array_nuc_count += 1

        create_monomer_topology(self.sequence, f'array_DNA_{NucleosomeArray._array_counter}', chain_type='DNA', topology=array_topology)
        NucleosomeArray._array_counter += 1

        return array_topology

    def generate_initial_coords(self, from_CG=None):
        """
        Generates initial coordinates for the nucleosome array.

        The coordinates are generated by calculating positions for both the nucleosomes and the associated DNA.

        Returns:
            ndarray: Initial 3D coordinates for the nucleosome array.
        """

        if from_CG is None:
            from_CG = self.from_CG  # Use the instance attribute if no argument is provided
        
        return generate_array_positions(self.nucleosome_id_sequence, self.DNA_sequence, self.dyad_positions, from_CG)
     
def create_monomer_topology(sequence, chain_id, chain_type, globular_indices=None, topology=None):
    """
    Creates an OpenMM Topology object for the given biomolecular sequence.

    This function builds the topology of a biomolecule by adding residues 
    and atoms to the topology. It can either create a new Topology object or add to an existing one.

    Args:
        sequence (str): The sequence of the biomolecule (e.g., protein or nucleic acids).
        chain_id (str): Identifier for the biomolecular chain in the topology.
        chain_type (str): Type of the chain ('prt' for protein, 'DNA' for DNA).
        globular_indices (list, optional): List of indices that represent globular regions of the sequence. Defaults to None.
        topology (Topology, optional): An existing OpenMM Topology object to add the chain to. If not provided, a new Topology will be created.

    Returns:
        Topology: The OpenMM Topology object representing the biomolecule.
    """
    
    chain_dict = {
        'prt': ['Cu', 'p'],
        'DNA': ['Au', 'd']
    }

    element_symbol = chain_dict[chain_type][0]
    prefix = chain_dict[chain_type][1]

    if not topology:
        topology = app.Topology()

    chain = topology.addChain(id=chain_id)

    if globular_indices is None:
        globular_indices = []

    flattened_globular_indices = [index for sublist in globular_indices for index in sublist]

    for res_id, residue_name in enumerate(sequence):
        if res_id in flattened_globular_indices:
            residue = topology.addResidue(prefix + residue_name, chain)
            topology.addAtom(prefix + residue_name, app.Element.getBySymbol('Pt'), residue)
        else:
            residue = topology.addResidue(prefix + residue_name, chain)
            topology.addAtom(prefix + residue_name, app.Element.getBySymbol(element_symbol), residue)

    # Special handling for DNA: Add extra particles for phosphate beads
    if chain_type == 'DNA':
        for _ in range(len(sequence) - 2): # 1 fewer phosphates than C1 per strand :L
            residue = topology.addResidue(prefix + 'P', chain)
            topology.addAtom(prefix + 'P', app.Element.getBySymbol('P'), residue)

    return topology
