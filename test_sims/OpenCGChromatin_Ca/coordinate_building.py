import numpy as np
import openmm.app as app
import openmm as mm
import openmm.unit as unit
import os
import random

from .constants import NUCLEOSOME_DATA, DATA_DIR

def generate_spiral_coords(N, spacing=0.381):
    """
    Generates a spiral configuration of N coordinates in 3D space, often used 
    for initializing coarse-grained biomolecular models.

    Args:
        N (int): The number of points to generate.
        spacing (float, optional): The spacing between adjacent points. Default is 0.381 nm.

    Returns:
        ndarray: An N x 3 array of 3D coordinates representing the spiral, centered around the origin.
    """
    theta = np.sqrt(np.arange(N) / float(N)) * 2 * np.pi
    r = spacing * np.sqrt(np.arange(N))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.linspace(-N * spacing / 2, N * spacing / 2, N)
    
    points = np.column_stack((x, y, z))
    
    return points - np.mean(points, axis=0)

def parse_pdb(pdb_file, globular_indices):
    """
    Parses a PDB file to extract coarse-grained (CG) coordinates based on globular regions and returns
    the sequence of the protein. Residues in globular domains are mapped to heavy-atom centre of geometry,
    residues in intrinsically-disordered domains are mapped to Ca.

    Args:
        pdb_file (str): Path to the PDB file to parse.
        globular_indices (list): List of lists containing indices of globular regions.

    Returns:
        tuple:
            - CG_coords (ndarray): N x 3 array of CG coordinates, centered around the origin.
            - pdb_sequence (str): The sequence of the protein extracted from the PDB file.
    """
    
    pdb = app.PDBFile(pdb_file)
    topology = pdb.topology
    aa_coords = pdb.positions

    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    residues = [residue for residue in topology.residues() if residue.name in aa_map.keys()]
    pdb_sequence = ''.join([aa_map[residue.name] for residue in residues])

    all_globular_indices = [index for domain in globular_indices for index in domain]
    
    CG_coords = []
    for index, residue in enumerate(residues):
        '''
        if index in all_globular_indices:
            heavy_atom_indices = [atom.index for atom in residue.atoms() if 'H' not in atom.name]
            CoG = np.mean([aa_coords[i]/unit.nanometer for i in heavy_atom_indices], axis=0)
            CG_coords.append(CoG)
        else:
        '''
        for atom in residue.atoms():
            if atom.name == 'CA':  
                CG_coords.append(aa_coords[atom.index]/unit.nanometer)
                break
    
    CG_coords = np.array(CG_coords)
    
    return CG_coords - np.mean(CG_coords, axis=0), pdb_sequence

def generate_array_positions(nuc_id_sequence, sequence, dyads_position, from_CG=True):
    """
    Generates 3D positions for an array of nucleosomes and DNA based on the provided nucleosome IDs, DNA sequence, and dyad positions.

    Args:
        nuc_id_sequence (list): List of nucleosome IDs in the array.
        sequence (str): DNA sequence associated with the array.
        dyads_position (list): List of dyad positions for the nucleosomes.
        from_CG (bool, optional): If True, use coarse-grained nucleosome structures for position generation. Defaults to True.

    Returns:
        ndarray: An N x 3 array of 3D coordinates (in nanometers) for the nucleosomes and DNA in the array.
    """
    N = len(sequence)

    if len(dyads_position) > 0:
        dyads_diff = np.diff(dyads_position)
        if np.any(dyads_diff < 147):
            raise ValueError(f'Some dyads are too close: {[f"Nucleosome {i} and {i+1}, with dyads at {dyads_position[i]} and {dyads_position[i+1]}, are at distance {dyads_diff[i]} < 147" for i in np.where(dyads_diff < 147)[0]]}')
    
    nucleosomal_dna = []
    nucleosome_cores = []

    for i, nucleosome_id in enumerate(nuc_id_sequence):
        nuc_data = NUCLEOSOME_DATA[nucleosome_id]
        if nuc_data is None:
            raise ValueError(f"Nucleosome ID {nucleosome_id} not found in NUCLEOSOME_DATA")

        aa_pdb_file_path = DATA_DIR + nuc_data['aa_pdb_path']
        IDRs = nuc_data['IDRs']
        cg_pdb_file_path = DATA_DIR + nuc_data['cg_pdb_path']
        N_nuc = nuc_data['N']

        if from_CG == True:
            reference_core, reference_nucleosomal_dna = parse_cg_nucleosome(cg_pdb_file_path)
        else:
            reference_core, _ = parse_pdb(aa_pdb_file_path, [[i for i in range(N_nuc) if not any(i in IDR for IDR in IDRs)]])
            reference_nucleosomal_dna = parse_nucleosomal_DNA(aa_pdb_file_path)

        nucleosomal_dna.append(reference_nucleosomal_dna)
        nucleosome_cores.append(reference_core)

        # Early return if sequence length matches reference nucleosomal DNA
        if 2 * len(sequence) == len(reference_nucleosomal_dna):
            positions = np.concatenate((reference_core, reference_nucleosomal_dna), axis=0)
            excluded_particles = calculate_virtual_sites(reference_nucleosomal_dna)
            positions = np.concatenate((positions, excluded_particles), axis=0)
            return positions - np.mean(positions, axis=0)

    dna_to_create = find_linker_regions(N, dyads_position)
    linker_dna = [generate_linear_DNA_coords(dna_i[1] - dna_i[0] + 1, linker=True) for dna_i in dna_to_create]

    all_dna = [linker_dna[0]]
    all_cores = []
    
    for i in range(len(dyads_position)):
        current_linker_dna = all_dna[-1]
        next_nucl_dna = nucleosomal_dna[i]

        next_linker_dna = linker_dna[i+1]
        next_nucl_core = nucleosome_cores[i]
    
        isometry = find_dna_isometry(current_linker_dna, next_nucl_dna)
        next_nucl_dna_t = apply_isometry(isometry, next_nucl_dna)
        all_dna.append(next_nucl_dna_t)
    
        isometry2 = find_dna_isometry(all_dna[-1], next_linker_dna)
        next_linker_dna_t = apply_isometry(isometry2, next_linker_dna)
        all_dna.append(next_linker_dna_t)
    
        R, t = isometry
        next_nucl_core_t = np.dot(R, next_nucl_core.T).T + t
        all_cores.append(next_nucl_core_t)

    # Join all DNA segments and nucleosome cores into a single array
    dna = join_dna(all_dna)
    positions = []

    for core_ca in all_cores:
        positions.extend(core_ca)

    for dna_pos in dna:
        positions.append(dna_pos)

    positions = np.array(positions)

    virtual_sites = calculate_virtual_sites(dna)
 
    positions = np.concatenate((positions, virtual_sites), axis=0)

    return positions - np.mean(positions, axis=0)
'''
def calculate_virtual_sites(C1_atoms):
    """
    Calculates the initial excluded volume virtual site positions by averaging the coordinates of corresponding C1' atoms from complementary DNA strands.

    Args:
        C1_atoms (ndarray): N x 3 array of C1 atom coordinates for the DNA strands.

    Returns:
        ndarray: M x 3 array of virtual site coordinates, where M is the number of virtual sites.
    """
    virtual_sites = []
    N = len(C1_atoms)

    for i in range(N-2):
        virtual_site = C1_atoms[i]
        virtual_sites.append(virtual_site)

    return np.array(virtual_sites)
'''
import numpy as np

def compute_virtual_site(r1, r2, r3, local_position):
    """
    Computes a virtual site position using a local coordinate transformation.
    
    Args:
        r1, r2, r3 (ndarray): 1x3 arrays corresponding to the coordinates of the three reference atoms.
            Here r1 is used as the origin, r2 defines the x direction, and r3 defines the y direction.
        local_position (ndarray): 1x3 array of local coordinates (Lx, Ly, Lz) for the virtual site.
        
    Returns:
        ndarray: 1x3 virtual site coordinate.
    """
    # Origin is given by r1 (consistent with origin_weights = [1, 0, 0])
    origin = r1
    
    # Define the x-axis: direction from r1 to r2
    x_vec = r2 - r1
    if np.linalg.norm(x_vec) == 0:
        raise ValueError("Zero length vector encountered for x-axis calculation.")
    x_hat = x_vec / np.linalg.norm(x_vec)
    
    # Define the y-axis: direction from r1 to r3, orthogonalized against x_hat
    y_vec = r3 - r1
    y_vec_ortho = y_vec - np.dot(y_vec, x_hat) * x_hat
    if np.linalg.norm(y_vec_ortho) == 0:
        raise ValueError("Zero length vector encountered for y-axis calculation after orthogonalization.")
    y_hat = y_vec_ortho / np.linalg.norm(y_vec_ortho)
    
    # Compute the virtual site position using the local coordinates.
    # Note: The z component is not used here since local_position[2] == 0.
    v_site = origin + local_position[0]*x_hat + local_position[1]*y_hat
    
    return v_site

def calculate_virtual_sites(C1_atoms):
    """
    Calculates initial coordinates for phosphate virtual sites in DNA strands.
    
    This function computes the initial positions for phosphate virtual sites by applying a local 
    coordinate transformation that is consistent with OpenMM's definition for phosphate virtual sites.
    The transformation uses three reference C1 atoms per site to define a local frame:
    
      - For the first strand:
          * r1 = s1_c1[i]
          * r2 = s1_c1[i+1]
          * r3 = s2_c1[i]
      
      - For the complementary strand:
          * r1 = s2_c1[-(i+1)]
          * r2 = s2_c1[-(i+2)]
          * r3 = s1_c1[-(i+1)]
    
    The local displacement used is:
    
          local_position = (-0.5, -0.3, 0.0)
    
    which matches the coordinates OpenMM will set when calling setVirtualSite for phosphate atoms.
    
    The input array `C1_atoms` is assumed to be of shape (N, 3) with an even number of atoms,
    where the first half corresponds to one DNA strand and the second half to its complement.
    
    To match the phosphate assignment routine—which assigns the highest-index phosphates with 
    the highest-index C1 atoms—the virtual sites computed for the complementary strand (strand 2)
    are reversed in order.
    
    Args:
        C1_atoms (ndarray): N x 3 array of C1 atom coordinates.
        
    Returns:
        ndarray: Array of phosphate virtual site coordinates in the same order as they are assigned.
    """
    # Check that the number of atoms is even.
    if len(C1_atoms) % 2 != 0:
        raise ValueError("The number of C1 atoms must be even to split into two strands.")
    
    num_c1 = len(C1_atoms) // 2

    # Split into two strands consistent with the VS definition:
    #   Strand 1: second half; Strand 2: reversed first half.
    s1_c1 = C1_atoms[:num_c1]
    s2_c1 = C1_atoms[num_c1:][::-1]
    
    # Define the local parameters (same as in set_phosphate_virtual_sites)
    local_position = np.array([-0.5, -0.3, 0.0])
    
    # For strand 1: compute virtual sites in natural order.
    num_sites = num_c1 - 1  # using i and i+1 requires at least two atoms per strand
    vs_strand1 = []
    for i in range(num_sites):
        # For strand 1: r1 = s1_c1[i], r2 = s1_c1[i+1], r3 = s2_c1[i]
        vs = compute_virtual_site(s1_c1[i], s1_c1[i+1], s2_c1[i], local_position)
        vs_strand1.append(vs)
    
    # For strand 2: compute virtual sites; these are computed with decreasing indices.
    vs_strand2 = []
    for i in range(num_sites):
        # For strand 2: r1 = s2_c1[-(i+1)], r2 = s2_c1[-(i+2)], r3 = s1_c1[-(i+1)]
        vs = compute_virtual_site(s2_c1[-(i+1)], s2_c1[-(i+2)], s1_c1[-(i+1)], local_position)
        vs_strand2.append(vs)
    
    # Reverse the strand 2 virtual sites so that high-index phosphates (assigned using negative indexing)
    # end up matched with high-index C1 atoms, consistent with the assignment routine.
    vs_strand2.reverse()
    
    # Concatenate the two lists so that the output ordering matches that used by the virtual site assignment.
    virtual_sites = vs_strand1 + vs_strand2
    
    return np.array(virtual_sites)

def generate_linear_DNA_coords(N, linker=False):
    """
    Generates 3D coordinates for a linear double-stranded DNA segment.

    Args:
        N (int): Number of base pairs to generate coordinates for.
        linker (bool, optional): If True, generates coordinates for linker DNA as part of a longer strand. 
            If False, generates coordinates for a stand-alone double-stranded DNA segment. Defaults to False.


    Returns:
        ndarray: An N x 3 array of DNA coordinates, including virtual sites if not in linker mode.
    """
    W = np.full((N, 3), [2.308, -5.333, -0.452])  
    C = np.full((N, 3), [2.308, 5.333, 0.452])   

    rise = 3.4  # Rise per base pair in nm
    twist = 35 * np.pi / 180.0  # Twist per base pair in radians

    W[:, 2] += np.arange(N) * rise
    C[:, 2] += np.arange(N) * rise
    twist_array = np.arange(N) * twist

    rot = np.array([[np.cos(twist_array), -np.sin(twist_array)], [np.sin(twist_array), np.cos(twist_array)]])
    rot = np.transpose(rot, (2, 1, 0))

    W[:, :2] = np.einsum('ijk,ij->ik', rot, W[:, :2])
    C[:, :2] = np.einsum('ijk,ij->ik', rot, C[:, :2])

    points = np.concatenate((W, C[::-1]), axis=0) / 10 

    if not linker:
        excluded_particles = calculate_virtual_sites(points)
        points = np.concatenate((points, excluded_particles), axis=0)

    return points - np.mean(points, axis=0)

def find_linker_regions(N, dyads_position):
    """
    Identifies linker regions in the DNA sequence based on the positions of nucleosome dyads.

    Args:
        N (int): Total length of the DNA sequence.
        dyads_position (list): List of positions of the nucleosome dyads in the DNA sequence.

    Returns:
        list: A list of tuples where each tuple contains the start and end positions of a linker region.
    """
    dyads_position = np.array(dyads_position)

    linker_begins = np.append([0], dyads_position + 74)
    linker_ends = np.append(dyads_position - 74, [N - 1])

    linker_regions = [(b, e) for b, e in zip(linker_begins, linker_ends)]

    return linker_regions

def find_isometry(target_coords, starting_coords):
    """
    Finds the rotation and translation needed to align the starting coordinates to the target coordinates.

    Args:
        target_coords (ndarray): M x 3 array of target coordinates.
        starting_coords (ndarray): M x 3 array of starting coordinates.

    Returns:
        tuple: A tuple containing:
            - R (ndarray): 3 x 3 rotation matrix.
            - Trans (ndarray): 1 x 3 translation vector.
    """
    
    if target_coords.shape[0] < 4 or target_coords.shape[1] != 3 or \
       starting_coords.shape[0] < 4 or starting_coords.shape[1] != 3:
        raise ValueError("Not a viable isometry. Input coordinates must have at least 4 points in 3D space.")

    target2 = target_coords - target_coords.mean(axis=0)
    starting2 = starting_coords - starting_coords.mean(axis=0)

    H = starting2.T @ target2
    U, S, V = np.linalg.svd(H)
    R = V.T @ U.T

    Trans = -R @ starting_coords.mean(axis=0) + target_coords.mean(axis=0)

    return R, Trans

def apply_isometry(isometry, starting_coords):
    """
    Applies an isometry (rotation and translation) to a set of starting coordinates.

    Args:
        isometry (tuple): A tuple containing:
            - R (ndarray): 3 x 3 rotation matrix.
            - Trans (ndarray): 1 x 3 translation vector.
        starting_coords (ndarray): N x 3 array of starting coordinates.

    Returns:
        ndarray: N x 3 array of transformed coordinates.
    """
    R, Trans = isometry
    new_coords = (R @ starting_coords.T).T + Trans
    return new_coords

def find_dna_isometry(current_dna, next_dna):
    """
    Finds the isometry (rotation and translation) needed to align two nucleosomal DNA segments.

    Args:
        current_dna (ndarray): N x 3 array of coordinates for the current DNA segment.
        next_dna (ndarray): N x 3 array of coordinates for the next DNA segment.

    Returns:
        tuple: A tuple containing:
            - R (ndarray): 3 x 3 rotation matrix.
            - Trans (ndarray): 1 x 3 translation vector.
    """
    n1=len(current_dna)
    current_tetrad=current_dna[n1//2-2:n1//2+2]
    next_tetrad=next_dna[[0,1,-2,-1]]

    # Create glue
    glue=generate_linear_DNA_coords(4, linker=True)
    glue_start=glue[[0,1,-2,-1]]
    iso1=find_isometry(current_tetrad,glue_start)
    glue_t=apply_isometry(iso1, glue)

    # Transform next dna to nucl
    glue_t_end=glue_t[8//2-2:8//2+2]
    iso2=find_isometry(glue_t_end,next_tetrad)
    return iso2

def join_dna(dna_list):
    """
    Joins coordinates of multiple DNA segments into one continuous sequence.

    Args:
        dna_list (list): List of N x 3 arrays, each representing a segment of DNA coordinates.

    Returns:
        ndarray: A single array of DNA coordinates obtained by joining the input DNA segments.
    """
    new_dna = dna_list[0]

    for next_dna in dna_list[1:]:
        new_dna = join_2_dna(new_dna, next_dna)

    return new_dna

def join_2_dna(current_dna, next_dna):
    """
    Joins coordinates for two DNA segments at their midpoint.

    Args:
        current_dna (ndarray): N x 3 array of coordinates for the current DNA segment.
        next_dna (ndarray): N x 3 array of coordinates for the next DNA segment.

    Returns:
        ndarray: A single array of DNA coordinates obtained by joining the two segments at their midpoint.
    """
    new_dna = np.concatenate((current_dna[:len(current_dna) // 2], next_dna, current_dna[len(current_dna) // 2:]))
    return new_dna

def parse_nucleosomal_DNA(pdb_file):
    """
    Parses an all-atom PDB file to extract the coordinates of nucleosomal DNA atoms.

    This function extracts the C1' atoms from the DNA residues in the PDB file, can
    be used to obtain coarse-grained structures.

    Args:
        pdb_file (str): Path to the PDB file containing nucleosomal DNA.

    Returns:
        ndarray: An N x 3 array of nucleosomal DNA coordinates, centered at the origin.
    """
    pdb = app.PDBFile(pdb_file)
    topology = pdb.topology
    aa_coords = pdb.positions

    DNA_residues = [residue for residue in topology.residues() if 'D' in residue.name]

    DNA_coords = []
    for residue in DNA_residues:
        for atom in residue.atoms():
            if atom.name == "C1'": 
                DNA_coords.append(aa_coords[atom.index] / unit.nanometer)

    DNA_coords = np.array(DNA_coords)

    return DNA_coords - np.mean(DNA_coords, axis=0)


def parse_cg_nucleosome(pdb_file):
    """
    Parses a coarse-grained nucleosome PDB file to extract nucleosome and DNA coordinates.

    This function extracts both nucleosome and DNA C1 coordinates, and centers the coordinates at the origin.

    Args:
        pdb_file (str): Path to the PDB file containing coarse-grained nucleosome data.

    Returns:
        tuple: A tuple containing:
            - nuc_coords (ndarray): N x 3 array of nucleosome coordinates.
            - DNA_coords (ndarray): N x 3 array of DNA coordinates.
    """
    pdb = app.PDBFile(pdb_file)
    topology = pdb.topology
    pdb_coords = pdb.positions

    nuc_coords = []
    DNA_coords = []
    
    for atom in topology.atoms():
        if 'dP' not in atom.name and 'Ex.' not in atom.name: # Phosphates are added at a later stage and have coordinates defined based on C1 positions, do not need to get from file.
            if 'd' in atom.name:   
                DNA_coords.append(pdb_coords[atom.index] / unit.nanometer)
            else:                  
                nuc_coords.append(pdb_coords[atom.index] / unit.nanometer)

    nuc_coords = np.array(nuc_coords)
   
    DNA_coords = np.array(DNA_coords)

    return nuc_coords - np.mean(nuc_coords, axis=0), DNA_coords - np.mean(DNA_coords, axis=0)
