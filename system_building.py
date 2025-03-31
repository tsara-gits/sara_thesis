import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from scipy.spatial import KDTree
import os
import copy
import warnings

from .constants import DATA_DIR, NUCLEOSOME_DATA, TETRAMER_DATA

def get_harmonic_bonds(positions, topology, globular_indices_dict, dyad_positions, constraints='inner', IDR_d=0.381, IDR_k=8031.):
    """
    Generates a list of harmonic bonds for a given topology. The function handles nucleosome chains, DNA chains,
    and other biomolecules and assigns bonds between atoms based on their positions and chain type.

    Args:
        positions (ndarray): An array of atomic positions.
        topology (Topology): An OpenMM topology object containing the chains and atoms of the system.
        globular_indices_dict (dict): A dictionary mapping chain IDs to lists of globular domain indices.
        dyad_positions (list): A list of dyad positions for nucleosomes.
        constraints (str, optional): Type of nucleosome-DNA constraint ('none', 'dyad', 'inner', 'breathing', 'all'). Default is 'inner'.
        IDR_d (float, optional): The equilibrium distance for harmonic bonds in intrinsically disordered regions (IDRs). Default is 0.381.
        IDR_k (float, optional): The force constant for harmonic bonds in IDRs. Default is 8031.
    
    Returns:
        list: A list of harmonic bonds, where each bond is represented as a tuple (atom1, atom2, distance, force constant).
    """
    
    bonds = []

    all_atoms = list(topology.atoms())  # Fetch all atoms in topology once
    atom_tree = KDTree(positions)

    chains = list(topology.chains())  # Fetch chains only once

    nucleosome_chain_array_indices = []
    for chain in chains:
        if 'nuc' in chain.id:
            split_id = chain.id.split('_')
            if len(split_id) > 2:
                nucleosome_chain_array_indices.append(int(split_id[2]))
            else:
                nucleosome_chain_array_indices.append(-1)
    
    array_dna_chains = [chain for chain in chains if 'array_DNA' in chain.id]  # Match DNA chains with nucleosomes
    # Iterate over all chains in topology
    for i, chain in enumerate(chains):
        chain_atoms = list(chain.atoms())

        if 'nuc' in chain.id:  # Nucleosome chains
            split_id = chain.id.split('_')
            nucleosome_id = split_id[1]
            array_idx = int(split_id[2]) if len(split_id) > 2 else -1
            nucleosome_idx = int(split_id[3]) if len(split_id) > 2 else -1

            if dyad_positions and array_idx != -1:   
                dyad_position = dyad_positions[array_idx][nucleosome_idx]
                dna_chain = array_dna_chains[array_idx]  # Get matching DNA chain
            else:
                dyad_position = None
                dna_chain = None
            # Get nucleosome harmonic bonds
            nuc_bonds = get_nucleosome_harmonic_bonds(chain_atoms, all_atoms, atom_tree, nucleosome_id, dyad_position, dna_chain, constraints)
            bonds.extend(nuc_bonds)

        else:  # Handle IDP and MDPs
            
            if 'd' in chain_atoms[0].name:  # Skip DNA chains
                continue

            chain_id = chain.id
            globular_indices_list = globular_indices_dict.get(chain_id, [])  # Use default empty list if not found

            # Identify IDR indices
            all_globular_indices = [i for domain in globular_indices_list for i in domain]
            IDR_indices = [i for i in range(len(chain_atoms)) if i not in all_globular_indices]

            # Add bonds for IDR regions
            for i in range(len(chain_atoms) - 1):
                if i in IDR_indices or i + 1 in IDR_indices:
                    bonds.append((chain_atoms[i], chain_atoms[i + 1], IDR_d, IDR_k))

            # Add ENM bonds for globular regions
            for globular_indices in globular_indices_list:
                ENM_atoms = [chain_atoms[i] for i in globular_indices]
                ENM_bonds = get_ENM_bonds(atom_tree, ENM_atoms, all_atoms)
                bonds.extend(ENM_bonds)

    return bonds

def get_nucleosome_harmonic_bonds(chain_atoms, all_atoms, atom_tree, nucleosome_id, dyad_position, dna_chain, constraints='inner', IDR_d=0.381, IDR_k=8031.):
    """
    Generates harmonic bonds for a nucleosome chain based on its IDRs and terminal information. Optionally adds harmonic bonds 
    between nucleosomal DNA and the nucleosome based on the specified constraints.

    Args:
        chain_atoms (list): List of atoms in the nucleosome chain.
        all_atoms (list): List of all atoms in the system.
        atom_tree (KDTree): KDTree of atomic positions for efficient neighbor lookup.
        nucleosome_id (str): Identifier for the nucleosome.
        dyad_position (int): Dyad position for the nucleosome.
        dna_chain (list): The associated DNA chain for the nucleosome.
        constraints (str, optional): Type of nucleosome-DNA constraint ('dyad', 'inner', 'breathing', 'all'). Default is 'inner'.
        IDR_d (float, optional): The equilibrium distance for harmonic bonds in IDRs. Default is 0.381.
        IDR_k (float, optional): The force constant for harmonic bonds in IDRs. Default is 8031.

    Returns:
        list: A list of harmonic bonds, where each bond is represented as a tuple (atom1, atom2, distance, force constant).
    """
    
    bonds = []

    # Retrieve nucleosome data from the global dictionary
    nucleosome_data = NUCLEOSOME_DATA[nucleosome_id]
    IDRs = nucleosome_data['IDRs']
    terminal_info = nucleosome_data['terminal_info']

    # Generate bonds within the IDRs
    for index, IDR in enumerate(IDRs):
        # Bonds within the IDR
        bonds.extend([(chain_atoms[IDR[i]], chain_atoms[IDR[i+1]], IDR_d, IDR_k) for i in range(len(IDR) - 1)])

        # Bonds to connect the terminal ends
        terminal_bond = (chain_atoms[IDR[0]], chain_atoms[IDR[0] - 1]) if terminal_info[index] == 'C' else (chain_atoms[IDR[-1]], chain_atoms[IDR[-1] + 1])
        bonds.append((*terminal_bond, IDR_d, IDR_k))

    # Collect all indices of IDR atoms
    all_IDR_indices = [index for IDR in IDRs for index in IDR]

    # Select atoms outside the IDRs for Elastic Network Model (ENM)
    ENM_atoms = [atom for i, atom in enumerate(chain_atoms) if i not in all_IDR_indices]

    # Add bonds between nucleosome and DNA based on constraints
    if (dna_chain is not None) and (dyad_position is not None):
        dna_indices = []
        if constraints == 'dyad':
            dna_indices = range(dyad_position - 1, dyad_position + 1)  # Dyad DNA
        elif constraints == 'inner':
            dna_indices = range(dyad_position - 36, dyad_position + 37)  # Inner turn DNA
        elif constraints == 'more_breathing':
            dna_indices = range(dyad_position - 53, dyad_position + 54)  # Nucleosomal DNA w/ 20 base pairs either side left out
        elif constraints == 'breathing':
            dna_indices = range(dyad_position - 63, dyad_position + 64)  # Nucleosomal DNA w/ 10 base pairs either side left out
        elif constraints == 'all':
            dna_indices = range(dyad_position - 73, dyad_position + 74)  # Nucleosomal DNA

        # Add corresponding DNA atoms to ENM_atoms
        dna_atoms = [atom for i, atom in enumerate(dna_chain.atoms()) if 'P' in atom.name]
        
        for index in dna_indices:
            ENM_atoms.append(dna_atoms[index])
            ENM_atoms.append(dna_atoms[-(index+1)])

    ENM_bonds = get_ENM_bonds(atom_tree, ENM_atoms, all_atoms, cutoff=0.75)
    bonds.extend(ENM_bonds)

    return bonds

def get_ENM_bonds(atom_tree, ENM_atoms, all_atoms, cutoff=0.75, k=8031.):
    """
    Generates a list of Elastic Network Model (ENM) bonds for a given set of atoms based on a distance cutoff.
    Excludes any DNA-DNA interactions, as ENM bonds are intended for protein or protein-DNA interactions.

    Args:
        atom_tree (KDTree): KDTree of atomic positions for efficient neighbor lookup.
        ENM_atoms (list): List of atoms to consider for ENM bonds.
        all_atoms (list): List of all atoms in the system.
        cutoff (float, optional): Distance cutoff for considering a bond (in nm). Default is 0.75 nm.
        k (float, optional): Force constant for the ENM bonds. Default is 8031.
    
    Returns:
        list: A list of ENM bonds, where each bond is represented as a tuple (atom1, atom2, distance, force constant).
    """
    
    bonds = []
    num_atoms = len(ENM_atoms)
    
    # Precompute positions of ENM atoms from the KDTree
    ENM_positions = {atom.index: atom_tree.data[atom.index] for atom in ENM_atoms}
    
    # Iterate through each atom in ENM_atoms
    for i in range(num_atoms):
        atom1 = ENM_atoms[i]
        atom1_pos = ENM_positions[atom1.index]
        
        # Query nearby atoms within the cutoff distance
        nearby_indices = atom_tree.query_ball_point(atom1_pos, cutoff)
        
        # Check each nearby atom and form a bond if it's not DNA-DNA
        for j in nearby_indices:
            if j != atom1.index and j in ENM_positions.keys():  # Exclude self
                atom2 = all_atoms[j]
                
                # exclude DNA-DNA bonds
                if not ('d' in atom1.name and 'd' in atom2.name):
                    atom2_pos = atom_tree.data[j]
                    r = np.linalg.norm(atom1_pos - atom2_pos)
                    bonds.append((atom1, atom2, r, k))

    return bonds

def seqtrans(sequence):
    """
    Reverse a DNA sequence and translate each nucleotide to its complementary base.
    
    Args:
        sequence (str): The input DNA sequence containing 'A', 'C', 'G', and 'T'.
        
    Returns:
        str: The reversed and complementary DNA sequence.
    """
    complement_map = str.maketrans({'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'})
    

    return sequence[::-1].translate(complement_map)

def number_tetramers(sequence):
    """
    Assigns a unique numerical identifier to each tetramer (four-nucleotide sequence) in the given DNA sequence.
    This function takes into account complementary sequences and assigns positive or negative values
    based on the original or complementary tetramer.

    Args:
        sequence (str): The input DNA sequence containing 'A', 'C', 'G', and 'T'.
    
    Returns:
        list: A list of tuples, where each tuple contains:
              - The numerical value assigned to the tetramer.
              - A shift factor (1 for original, -1 for complementary).
    """
    
    # Create a dictionary of all unique tetramers (4-nucleotide combinations)
    NTetramers = {a + b + c + d: i for i, (a, b, c, d) in enumerate(
        [(a, b, c, d) for a in 'ACGT' for b in 'ACGT' for c in 'ACGT' for d in 'ACGT']
    )}
    
    # Initialize an array to store the tetramer values
    Tetramers = [None] * len(NTetramers)
    i = 1
    
    # Assign positive or negative values based on the original or complementary tetramer
    for element in NTetramers:
        rev_complement = seqtrans(element)
        if Tetramers[NTetramers[rev_complement]] is None:
            Tetramers[NTetramers[element]] = i
            i += 1
        else:
            Tetramers[NTetramers[element]] = -Tetramers[NTetramers[rev_complement]]
    
    # Segment the input sequence into tetramers and map them to numerical values
    segmented_sequence = [sequence[i:i+4] for i in range(len(sequence) - 3)]
    segmented_numbers = np.vectorize(lambda x: Tetramers[NTetramers[x]])(np.array(segmented_sequence))
    
    # Create the shift factor based on positive or negative values
    shift = (segmented_numbers > 0) * 2 - 1
    segmented_numbers *= shift
    
    # Return a list of (tetramer value, shift factor) tuples
    return list(zip(segmented_numbers, shift))

def apply_bond_units(bond_list, k_coeff=1.):
    """
    Apply OpenMM bond units to a given bond list, scaling the force constants by the provided coefficient.
    
    Args:
        bond_list (list): A list of bond parameters to which units will be applied.
                         - bond_list[0]: Distance in angstroms.
                         - bond_list[1-3]: Force constants for bond energy terms.
        k_coeff (float, optional): Scaling coefficient for the force constants. Default is 1.0.

    Returns:
        list: The bond list with appropriate OpenMM units applied to each parameter.
    """ 
    bond_list = copy.deepcopy(bond_list)
    
    bond_list[0] *= unit.angstrom
    bond_list[1] *= k_coeff * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
    bond_list[2] *= k_coeff * unit.kilocalorie_per_mole / (unit.angstrom ** 3)
    bond_list[3] *= k_coeff * unit.kilocalorie_per_mole / (unit.angstrom ** 4)

    return bond_list

def apply_angle_units(angle_list, k_coeff=1.):
    """
    Apply OpenMM angle units to a given angle list, scaling the force constants by the provided coefficient.

    Args:
        angle_list (list): A list of angle parameters to which units will be applied.
                           - angle_list[0]: Angle in degrees.
                           - angle_list[1-3]: Force constants for angle energy terms.
        k_coeff (float, optional): Scaling coefficient for the force constants. Default is 1.0.

    Returns:
        list: The angle list with appropriate OpenMM units applied to each parameter.
    """
    angle_list = copy.deepcopy(angle_list)
    
    angle_list[0] *= unit.degree
    angle_list[1] *= k_coeff * unit.kilocalorie_per_mole / (unit.radian ** 2)
    angle_list[2] *= k_coeff * unit.kilocalorie_per_mole / (unit.radian ** 3)
    angle_list[3] *= k_coeff * unit.kilocalorie_per_mole / (unit.radian ** 4)

    return angle_list

def get_DNA_bonds_angles(topology, mechanics_k_coeff=1.0):
    """
    Generates bond and angle parameters for DNA chains in the given topology.

    This function calculates bonds and angles based on tetramers (four-nucleotide sequences) within DNA chains.
    It assigns bond and angle parameters using predefined data from `TETRAMER_DATA`, taking into account
    the reverse complement of the DNA sequence and adding necessary exclusions. Additionally, auxiliary bonds
    for n-n+4 and n-n+5 atom pairs are added.

    Args:
        topology (Topology): An OpenMM topology object that contains DNA chains, from which bonds and angles will be calculated.
        mechanics_k_coeff (float, optional): Scaling coefficient for mechanical force constants. Default is 1.0.
    
    Returns:
        tuple: A tuple containing:
            - bonds (list): A list of bonds, where each bond is a list containing two atoms and their bond parameters.
            - angles (list): A list of angles, where each angle is a list containing three atoms and their angle parameters.
    """

    exclusion = 4*[0.] # Non-bonded exclusions applied using a bond with 0. for each parameter
    
    tetramer_bonds_angles, auxiliary_bonds = TETRAMER_DATA['tetramer_bonds_angles'], TETRAMER_DATA['auxiliary_bonds']
    
    bonds = []
    angles = []
    
    for chain in topology.chains():
        atoms = list(chain.atoms())
        
        if 'd' in atoms[0].name: # check if we have a DNA chain
            atoms = [atom for atom in atoms if atom.name != 'dP'] # phosphates are not included
            N = len(atoms) 
        
            sequence = 'G'+''.join([atom.name.replace('d','') for atom in atoms[:N//2]])+'C' #adding pseudo-bases to the sequence so termini are handled correctly
            
            tetramers = number_tetramers(sequence)  
        
            for i, tetramer in enumerate(tetramers): 

                tetnum, reversed_flag = tetramer
                is_reversed = (reversed_flag == -1)
                
                # look up the bond/angle parameters for this tetramer in nested dict
                bond_parameters = tetramer_bonds_angles[tetnum]['bonds']
                angle_parameters = tetramer_bonds_angles[tetnum]['angles']
        
                if i == 0:  # First pseudo-tetramer
                    i0, i1, i2, i3 = None, atoms[i], atoms[i+1], atoms[i+2]
                    j0, j1, j2, j3 = None, atoms[N-(i+1)], atoms[N-(i+2)], atoms[N-(i+3)]
                elif i == len(tetramers) - 1:  # Last pseudo-tetramer
                    i0, i1, i2, i3 = atoms[i-1], atoms[i], atoms[i+1], None
                    j0, j1, j2, j3 = atoms[N-(i)], atoms[N-(i+1)], atoms[N-(i+2)], None
                else:  # Regular tetramers
                    i0, i1, i2, i3 = atoms[i-1], atoms[i], atoms[i+1], atoms[i+2]
                    j0, j1, j2, j3 = atoms[N-(i)], atoms[N-(i+1)], atoms[N-(i+2)], atoms[N-(i+3)]
                                        
                tetramer_bonds = [
                                    [i1, i2, apply_bond_units(bond_parameters[1], k_coeff=mechanics_k_coeff)],
                                    [j1, j2, apply_bond_units(bond_parameters[2], k_coeff=mechanics_k_coeff)],
                                    [i0, j3, apply_bond_units(bond_parameters[3], k_coeff=mechanics_k_coeff)],
                                    [i0, j2, apply_bond_units(bond_parameters[4], k_coeff=mechanics_k_coeff)],
                                    [i1, j3, apply_bond_units(bond_parameters[5], k_coeff=mechanics_k_coeff)],
                                    [i1, j2, apply_bond_units(bond_parameters[6], k_coeff=mechanics_k_coeff)],
                                    [i1, j1, apply_bond_units(bond_parameters[7], k_coeff=mechanics_k_coeff)],
                                    [i2, j2, apply_bond_units(bond_parameters[8], k_coeff=mechanics_k_coeff)],
                                    [i2, j1, apply_bond_units(bond_parameters[9], k_coeff=mechanics_k_coeff)],
                                    [i2, j0, apply_bond_units(bond_parameters[10], k_coeff=mechanics_k_coeff)],
                                    [i3, j1, apply_bond_units(bond_parameters[11], k_coeff=mechanics_k_coeff)],
                                    [i3, j0, apply_bond_units(bond_parameters[12], k_coeff=mechanics_k_coeff)],
                                    [i0, i3, exclusion],
                                    [j0, j3, exclusion]
                                 ]
                    
                tetramer_angles = [
                                    [i0, i1, i2, apply_angle_units(angle_parameters[1], k_coeff=mechanics_k_coeff)],
                                    [i1, i2, i3, apply_angle_units(angle_parameters[2], k_coeff=mechanics_k_coeff)],
                                    [j0, j1, j2, apply_angle_units(angle_parameters[3], k_coeff=mechanics_k_coeff)],
                                    [j1, j2, j3, apply_angle_units(angle_parameters[4], k_coeff=mechanics_k_coeff)]
                                  ]
        
                # if is_reversed == True, reverse order of tetramer bonds 1 and 2 and all tetramer angles
                if is_reversed:
                    tetramer_bonds[0] = [i1, i2, apply_bond_units(bond_parameters[2], k_coeff=mechanics_k_coeff)]
                    tetramer_bonds[1] = [j1, j2, apply_bond_units(bond_parameters[1], k_coeff=mechanics_k_coeff)]
                    
                    
                    tetramer_angles = [
                                        [i0, i1, i2, apply_angle_units(angle_parameters[4], k_coeff=mechanics_k_coeff)],
                                        [i1, i2, i3, apply_angle_units(angle_parameters[3], k_coeff=mechanics_k_coeff)],
                                        [j0, j1, j2, apply_angle_units(angle_parameters[2], k_coeff=mechanics_k_coeff)],
                                        [j1, j2, j3, apply_angle_units(angle_parameters[1], k_coeff=mechanics_k_coeff)]
                                      ]

                tetramer_bonds = [bond for bond in tetramer_bonds if None not in bond]
                tetramer_angles = [angle for angle in tetramer_angles if None not in angle]
                                                      
                bonds.extend(tetramer_bonds)
                angles.extend(tetramer_angles)
        
            for i in range(N//2-4):
                i0, i4 = atoms[i], atoms[i+4]
                j0, j4 = atoms[N-(i+1)], atoms[N-(i+5)]
                
                bonds.extend([
                    [i0, j4, apply_bond_units(auxiliary_bonds[13], k_coeff=mechanics_k_coeff)],
                    [j0, i4, apply_bond_units(auxiliary_bonds[14], k_coeff=mechanics_k_coeff)],
                    [i0, i4, exclusion],
                    [j0, j4, exclusion]
                ])
            
            for i in range(N//2-5):
                i0, i5 = atoms[i], atoms[i+5]
                j0, j5 = atoms[N-(i+1)], atoms[N-(i+6)]
                
                bonds.extend([
                    [i0, j5, apply_bond_units(auxiliary_bonds[15], k_coeff=mechanics_k_coeff)],
                    [j0, i5, apply_bond_units(auxiliary_bonds[16], k_coeff=mechanics_k_coeff)],
                    [i0, i5, exclusion],
                    [j0, j5, exclusion]
                ])

    return bonds, angles

def set_phosphate_virtual_sites(system, topology):
    """
    Configures virtual sites for phosphate atoms in DNA chains within the given system and topology.
    
    This function identifies phosphate atoms ('dP') in each DNA chain, along with nearby C1 atoms, 
    and sets phosphates as local coordinate-based virtual sites, with their positions determined relative 
    to their corresponding C1 atoms.

    Args:
        system (System): The OpenMM System object where virtual sites will be set.
        topology (Topology): The OpenMM Topology associated with the system.
    """
    for chain in topology.chains():
        atoms = list(chain.atoms())
    
        if 'd' in atoms[0].name:

            phosphate_atoms = [atom for atom in atoms if atom.name == 'dP']
            c1_atoms = [atom for atom in atoms if atom.name in ['dA', 'dC', 'dG', 'dT']]
        
            num_c1_per_strand = len(c1_atoms) // 2
            num_p_per_strand = len(phosphate_atoms) // 2
        
            s1_c1 = c1_atoms[:num_c1_per_strand]
            s2_c1 = c1_atoms[num_c1_per_strand:][::-1]
        
            s1_p = phosphate_atoms[:num_p_per_strand]
            s2_p = phosphate_atoms[num_p_per_strand:]
        
            for i in range(num_p_per_strand):
                # Phosphate in the first strand
                p_i = s1_p[i]
                c1_i = s1_c1[i]
                c1_i_next = s1_c1[i + 1]
                c1_j = s2_c1[i]
        
                origin_weights = mm.Vec3(1.0, 0.0, 0.0)
                x_weights = mm.Vec3(-0.5, 0.5, 0.0) # p1 -> p2, bp axis
                y_weights = mm.Vec3(-0.5, 0.0, 0.5) # p1 -> p3, strand axis
                local_position = mm.Vec3(-0.5, -0.3, 0.0) 
        
                v1 = system.setVirtualSite(p_i.index, mm.LocalCoordinatesSite(c1_i.index, c1_i_next.index, c1_j.index,
                                                                         origin_weights, x_weights, y_weights,
                                                                         local_position))
                
                # Phosphate in the second strand
                p_i = s2_p[-(i+1)]
                c1_i = s2_c1[-(i+1)]
                c1_i_next = s2_c1[-(i+2)]
                c1_j = s1_c1[-(1 + i)]
                
                
                v2 = system.setVirtualSite(p_i.index, mm.LocalCoordinatesSite(c1_i.index, c1_i_next.index, c1_j.index,
                                                                         origin_weights, x_weights, y_weights,
                                                                         local_position))

def get_system(positions, topology, globular_indices_dict,
                dyad_positions, constraints='breathing',
                mechanics_k_coeff=1.20, overall_LJ_eps_scaling=0.75,
                 qP=1.0, qTail=1.0, qCore=1.0,
                  anchor_scaling=0.0, kappa_factor=1.0,
                    debye_length=.79365, coul_rc=None, CoMM_remover=False, periodic=True):
    """
    Constructs and returns an OpenMM System object based on the provided topology and particle positions.
    The function incorporates harmonic bonds, class2 style bonds and angles, an Asbaugh-Hatch Lennard-Jones (LJ) potential, a Coulomb potential, and
    an optional center-of-mass motion remover.
    Args:
        positions (ndarray): 3D array of atomic positions (in nanometers) for all particles in the system.
        topology (Topology): OpenMM Topology object that defines the structure of the system.
        globular_indices_dict (dict): Dictionary mapping chain IDs to lists of globular domain indices.
        dyad_positions (list): List of dyad positions associated with nucleosomes.
        constraints (str, optional): Defines the type of nucleosome-DNA constraint
                                        ('none', 'dyad', 'inner', 'breathing', 'all'). Defaults to 'breathing'.
        mechanics_k_coeff (float, optional): Scaling coefficient for mechanics-related bonds/angles. Default is 2.0.
        LJ_eps_scaling (float, optional): Scaling factor for the Lennard-Jones epsilon parameter. Default is 0.8.
        debye_length (float, optional): Debye length for screening in the Coulomb potential. Defaults to 0.79365.
        coul_rc (float, optional): Cutoff distance for the Coulomb potential in nanometers.
                                    If not provided, it defaults to 3.2 + 1 * debye_length.
        CoMM_remover (bool, optional): Whether to include a center-of-mass motion remover. Defaults to False.
        periodic (bool, optional): Whether to apply periodic boundary conditions. Defaults to True.
    Returns:
        System: An OpenMM System object with all forces and particle properties defined.
    """
    KH_PARAMETERS = np.loadtxt(os.path.join(DATA_DIR, 'KH_params.txt'))
    
    qAA=1.0
    mapping_dict =  {'pM': [131.2, 12, 0],
                     'pG': [57.05, 7, 0],
                     'pK': [128.2, 11, qAA],
                     'pT': [101.1, 16, 0],
                     'pR': [156.2, 1, qAA],
                     'pA': [71.08, 0, 0],
                     'pD': [115.1, 3, -qAA],
                     'pE': [129.1, 6, -qAA],
                     'pY': [163.2, 18, 0],
                     'pV': [99.07, 19, 0],
                     'pL': [113.2, 10, 0],
                     'pQ': [128.1, 5, 0],
                     'pW': [186.2, 17, 0],
                     'pF': [147.2, 13, 0],
                     'pS': [87.08, 15, 0],
                     'pH': [137.1, 8, 0.5*qAA],
                     'pN': [114.1, 2, 0],
                     'pP': [97.12, 14, 0],
                     'pC': [103.1, 4, 0],
                     'pI': [113.2, 9, 0],
                     'dA': [313.2, 20, 0],
                     'dC': [289.2, 21, 0],
                     'dG': [329.2, 22, 0],
                     'dT': [304.2, 23, 0],
                     'dP': [0.,    24, -qP]}
  
    system = mm.System()
    for atom in topology.atoms():
        system.addParticle(mapping_dict[atom.name][0])
   
    if constraints not in ('none', 'dyad', 'inner', 'more_breathing', 'breathing', 'all'):
        warnings.warn('Constraints option not recognised; defaulting to breathing.')
        constraints = 'breathing'
    
    harm_bonds = get_harmonic_bonds(positions, topology, globular_indices_dict, dyad_positions, constraints)
    bond_flag = True
    if len(list(topology.bonds())) > 0:
        bond_flag = False
   
    harm_potential = mm.HarmonicBondForce()
    for bond in harm_bonds:
        a1, a2, d, k = bond
        harm_potential.addBond(a1.index, a2.index, d, k)
        if bond_flag == True:
            topology.addBond(a1, a2)
   
    for chain in topology.chains():
        atoms = list(chain.atoms())
        if 'd' in atoms[0].name:
            phosphates = [atom for atom in atoms if 'P' in atom.name]
            n = len(phosphates)//2
            s1_phosphates, s2_phosphates = phosphates[:n], phosphates[n:]
            for i in range(n-1):
                topology.addBond(s1_phosphates[i], s1_phosphates[i+1])
                topology.addBond(s2_phosphates[i], s2_phosphates[i+1])
    system.addForce(harm_potential)
   
    c2_bond_string =  'K2*(r-r_0)^2 + K3*(r-r_0)^3 + K4*(r-r_0)^4'
    c2_angle_string = 'K2*(theta-theta_0)^2 + K3*(theta-theta_0)^3 + K4*(theta-theta_0)^4'
    c2_bond_potential = mm.CustomBondForce(c2_bond_string)
    c2_angle_potential = mm.CustomAngleForce(c2_angle_string)
    
    c2_bond_potential.addPerBondParameter('r_0')
    c2_bond_potential.addPerBondParameter('K2')
    c2_bond_potential.addPerBondParameter('K3')
    c2_bond_potential.addPerBondParameter('K4')

    c2_angle_potential.addPerAngleParameter('theta_0')
    c2_angle_potential.addPerAngleParameter('K2')
    c2_angle_potential.addPerAngleParameter('K3')
    c2_angle_potential.addPerAngleParameter('K4')
    
    DNA_bonds, DNA_angles = get_DNA_bonds_angles(topology, mechanics_k_coeff=mechanics_k_coeff)
    for bond in DNA_bonds:
        a1, a2, parameters = bond
        c2_bond_potential.addBond(a1.index, a2.index, parameters)
        if bond_flag == True and parameters[0] != 0.:
            topology.addBond(a1, a2)
    system.addForce(c2_bond_potential)

    for angle in DNA_angles:
        a1, a2, a3, parameters = angle
        c2_angle_potential.addAngle(a1.index, a2.index, a3.index, parameters)
    system.addForce(c2_angle_potential)

    ah_string = f'''
    step(rc-r)*select(step(r-2^(1/6)*sigma), term1, term2);
    
    term1 = 4*epsilon*lambda*((sigma/r)^12-(sigma/r)^6-shift);
    term2 = 4*epsilon*((sigma/r)^12-(sigma/r)^6-lambda*shift)+epsilon*(1-lambda);
    shift = (sigma/rc)^12-(sigma/rc)^6;
    rc = 3*sigma;
    
    epsilon = globular_scaling * {overall_LJ_eps_scaling} * KH_table(index1, index2, 0);

    globular_scaling = select((globular1+1)*(globular2+1), select(globular1*globular2, 0.7, select(globular1+globular2, sqrt(0.7), 1)), 1.0);

    sigma = KH_table(index1, index2, 1);
    lambda = KH_table(index1, index2, 2);
    '''
    ah_potential = mm.CustomNonbondedForce(ah_string)

    coulomb_string = '''
    (k_coulomb*q1*q2/(relative_permittivity*r))*exp(-kappa*r);
    '''

    coulomb_potential = mm.CustomNonbondedForce(coulomb_string)
    ah_potential.addPerParticleParameter('index')  # index
    ah_potential.addPerParticleParameter('globular')

    coulomb_potential.addPerParticleParameter('q')  # charge
    coulomb_potential.addGlobalParameter('k_coulomb', 138.935)
    coulomb_potential.addGlobalParameter('relative_permittivity', 80.0)
    coulomb_potential.addGlobalParameter('kappa', 1/debye_length)
    
    if periodic == True:
        ah_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        coulomb_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    else:
        ah_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        coulomb_potential.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        
    ah_potential.setCutoffDistance(2.5*unit.nanometer)
    ah_potential.setForceGroup(0)
    
    if coul_rc is None:
        coul_rc = 2. + 2.*debye_length

    coulomb_potential.setCutoffDistance(coul_rc*unit.nanometer)
    coulomb_potential.setForceGroup(1)

    for chain in topology.chains():
        atoms = list(chain.atoms())
        if 'nuc' in chain.id: # as above, nucleosomes require a bit more care
            nucleosome_id = chain.id.split('_')[1]
            nucleosome_data = NUCLEOSOME_DATA[nucleosome_id]
            IDRs = nucleosome_data['IDRs']
            terminal_info = nucleosome_data['terminal_info']
            N_terminal_indices = []
            C_terminal_indices = []
            for IDR, terminal in zip(IDRs, terminal_info):
                if terminal == 'N':
                    N_terminal_indices.append(IDR[-1])
                elif terminal == 'C':
                    C_terminal_indices.append(IDR[0])
            for i, atom in enumerate(atoms):
                anchor_condition = (atom.name == 'pR') & (atom.element.symbol == 'Pt') 
                
                protein = 1 if atom.element.symbol in ['Pt', 'Cu'] else 0
                dna = 0
                globular = 1 if atom.element.symbol == 'Pt' else 0

                index = mapping_dict[atom.name][1]
                charge = mapping_dict[atom.name][2]

                charge *= (qCore if atom.element.symbol == 'Pt' else qTail)

                ah_potential.addParticle([index, globular])
                coulomb_potential.addParticle([charge])
        else:
            for i, atom in enumerate(atoms):
                protein = 1 if atom.element.symbol in ['Pt', 'Cu'] else 0
                dna = 1 if atom.element.symbol in ['P'] else 0 # for now make it only P to see if it improves stability
                
                if atom.element.symbol == 'Pt':
                    globular = 1
                elif atom.element.symbol == 'Cu':
                    globular = 0
                else:
                    globular = -1
        
                index = mapping_dict[atom.name][1]
                charge = mapping_dict[atom.name][2]

                ah_potential.addParticle([index, globular])
                coulomb_potential.addParticle([charge])
                
  
    added_exclusions = set()
    for bond in topology.bonds():
        atom1, atom2 = bond[0], bond[1]
        # Create a sorted tuple to avoid order issues
        pair = tuple(sorted([atom1.index, atom2.index]))
        if pair in added_exclusions:
            continue

        sym1, sym2 = atom1.element.symbol, atom2.element.symbol
        add_exclusion = False

        # Exclude if a Cu atom is bonded to either Cu or Pt
        if (sym1 == 'Cu' and sym2 in ['Cu', 'Pt']) or (sym2 == 'Cu' and sym1 in ['Cu', 'Pt']):
            add_exclusion = True

        # Exclude if a P atom is bonded to another P atom
        elif sym1 == 'P' or sym2 == 'P':
            add_exclusion = True

        # Exclude if two Pt atoms are bonded and their initial distance is less than 0.5 nm
        elif sym1 == 'Pt' and sym2 == 'Pt':
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            distance = np.linalg.norm(pos1 - pos2)
            if distance < 0.5:
                add_exclusion = True

        if add_exclusion:
            ah_potential.addExclusion(atom1.index, atom2.index)
            coulomb_potential.addExclusion(atom1.index, atom2.index)
            added_exclusions.add(pair)

    KH_table = mm.Discrete3DFunction(25, 25, 3, KH_PARAMETERS)
    ah_potential.addTabulatedFunction('KH_table', KH_table)
    system.addForce(ah_potential)
    system.addForce(coulomb_potential)
    if CoMM_remover == True:
        system.addForce(mm.CMMotionRemover(1000))
    ranges = np.ptp(positions, axis=0)
    max_range_value = ranges[np.argmax(ranges)]
    box_length = max_range_value + 50
    box_vecs = [mm.Vec3(x=box_length, y=0.0, z=0.0), mm.Vec3(x=0.0, y=box_length, z=0.0), mm.Vec3(x=0.0, y=0.0, z=box_length)]*unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box_vecs)
    set_phosphate_virtual_sites(system, topology)
    return system

def get_minimized_system(positions, topology, globular_indices_dict, dyad_positions,
                         constraints='breathing', mechanics_k_coeff=1.20,
                         overall_LJ_eps_scaling=0.75, qP=1.0, qTail=0.9, qCore=1.0,
                         anchor_scaling=0.0, kappa_factor=1.0, debye_length=0.79365,
                         coul_rc=None, CoMM_remover=False, periodic=True):
    """
    Builds an OpenMM system using a two-step procedure.
    
    First, the system is built with constraints set to 'none' and energy minimized so that
    the virtual site (VS) positions relax to consistent coordinates. The minimized positions are
    then used to rebuild the system with the desired (non-'none') constraints, ensuring that the
    VS coordinates are properly initialized.
    
    Args:
        positions (ndarray): 3D array of initial atomic positions (in nanometers).
        topology (Topology): OpenMM Topology object.
        globular_indices_dict (dict): Mapping from chain IDs to globular domain indices.
        dyad_positions (list): List of dyad positions for nucleosomes.
        constraints (str, optional): Desired constraint type for the final system 
                                     (e.g. 'breathing', 'inner', etc.). Default is 'breathing'.
        mechanics_k_coeff, overall_LJ_eps_scaling, protein_DNA_LJ_eps_scaling, qP, qAA,
        anchor_scaling, kappa_factor, debye_length, coul_rc, CoMM_remover, periodic:
                                     Other parameters passed to get_system.
    
    Returns:
        tuple: (system_final, minimized_positions)
            system_final: The OpenMM System built using the minimized positions and desired constraints.
            minimized_positions: The positions after minimization.
    """
    # Step 1: Build an initial system with constraints='none'
    system_none = get_system(positions, topology, globular_indices_dict, dyad_positions, constraints='none')
    
    # Step 2: Minimize the energy using a simple integrator
    integrator = mm.VerletIntegrator(1)
    simulation = app.Simulation(topology, system_none, integrator)
    simulation.context.setPositions(positions)
    
    simulation.minimizeEnergy()
    
    # Extract the minimized positions from the simulation state.
    state = simulation.context.getState(getPositions=True)
    minimized_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    
    # Step 3: Build the final system with the desired configuration, including restraints, using the minimized positions.
    system_final = get_system(minimized_positions, topology, globular_indices_dict,
                            dyad_positions, constraints=constraints, 
                            mechanics_k_coeff=mechanics_k_coeff, overall_LJ_eps_scaling=overall_LJ_eps_scaling,
                            qP=qP,qTail=qTail, qCore=qCore, anchor_scaling=anchor_scaling, kappa_factor=kappa_factor,
                            debye_length=debye_length, coul_rc=coul_rc, CoMM_remover=CoMM_remover, periodic=periodic)
    
    return system_final, minimized_positions

