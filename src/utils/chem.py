# inspired by code from chem.py of the initial paper repository


import numpy as np
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


atomic_numbers = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
                  "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
                  "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,
                  "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
                  "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56}

def mol_from_jbonds_and_blocks(jbonds, blocks=None, block_smis=None):
    """
    params:
    jbonds: list of junction bonds in the molecule
    blocks: list of blocks in rdkit.Mols format
    blocks_smis: list of smiles strings for each block
    
    return: molecule in rdkit.Chem.rdchem.Mol form
    """
    
    # change junction bonds to numpy array
    jun_bonds = np.asarray(jbonds)

    # if not already in rdkit format change to it
    if blocks is None:
        if block_smis is not None:
            blocks = [Chem.MolFromSmiles(smi) for smi in block_smis]
        else:
            raise AssertionError("Cannot build molecule without frags or frag_smis!")
        
    # count number of frags
    nblocks = len(blocks)
    if nblocks == 0: return None, None
    
    # combine frags into a single molecule
    mol = blocks[0]
    for i in np.arange(1,nblocks):
        mol = Chem.CombineMols(mol, blocks[i])
    
    # get slices array for molecule
    # exclude last index
    block_startidx = np.concatenate([[0], np.cumsum([block.GetNumAtoms() for block in blocks])], 0)[:-1]

    # check if any junction bonds exists
    if jun_bonds.size == 0:
        mol_bonds = []
    else:
        # creates list of bonds
        # e.g.
        # jun_bonds = [[0,1,0,0],[0,2,1,1]]
        # frag_startidx = [0,6,8]
        # mol_bonds = [[0,6],[1,9]]
        mol_bonds = block_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]
    
    # make it possible to add bonds to molecule
    emol = Chem.EditableMol(mol)

    # add single covalent bonds between blocks using the obtained mol_bonds
    [emol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE) for bond in mol_bonds]

    # get molecule into Mol format (normal molecule)
    mol = emol.GetMol()
    atoms = list(mol.GetAtoms())

    # make space to add blocks
    # i.e. remove existing hydrogen atoms
    def _pop_H(atom):
        nh = atom.GetNumExplicitHs()
        if nh > 0: atom.SetNumExplicitHs(nh-1)

    # remove the hydrogen atoms for each bond
    [(_pop_H(atoms[bond[0]]), _pop_H(atoms[bond[1]])) for bond in mol_bonds]
    
    # sanitize mol
    Chem.SanitizeMol(mol)
    
    return mol, mol_bonds

# taken directly from https://github.com/GFNOrg/gflownet/blob/master/mols/utils/chem.py
def onehot(arr, num_classes, dtype=int):
    """
        params:
        arr: class indices
        num_classes: number of classes
        returns onehot encoded vector of shape (len(arr),num_classes)

        e.g.
        onehot([0,0,1,2],3) = [[1,0,0],[1,0,0],[0,1,0],[0,0,1]]
    """
    arr = np.asarray(arr, dtype=int)
    assert len(arr.shape) == 1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


# inspired heavily by code from https://github.com/GFNOrg/gflownet/blob/master/mols/utils/chem.py
def mol_to_graph_backend(atmfeat, bond, bondfeat):
    """"
    params:
    atmfeat: atom features (array of shape (natoms, 70))
    bond: list of bonds (array of shape (nbonds,2))
    bondfeat: list of bond types for each bond in onehot encoded format (array of shape (nbonds,4))

    convert to PyTorch geometric module graph
    """
    # number of atoms in molecule
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    # check if any bonds exist
    if bond.shape[0] > 0:
        # bond.T will have shape (2, nbonds)
        # e.g. the first bond goes from bond.T[0][0] to bond.T[1][0]
        # np.flipud(bond.T) simply flips the matrix upside down
        # e.g. bond = [[0,1],
        #              [1,2],
        #              [2,3]]
        # bond.T = [[0,1,2],
        #           [1,2,3]]
        # np.flipud(bond.T) = [[1,2,3],
        #                      [0,1,2]]
        # np.concatenate([bond.T, np.flipud(bond.T)], axis=1)=
        # [[0,1,2,1,2,3],
        #  [1,2,3,0,1,2]]
        # ensures that edges go both ways
        edge_index = torch.tensor(np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64)
        edge_attr = torch.tensor(np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32)
        # sorts the bonds afer indices
        edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
        # referring to the same example as before
        # [[0,1,2,1,2,3],
        #  [1,2,3,0,1,2]]
        # after coalesce
        # [[0,1,1,2,2,3],
        #  [1,0,2,1,3,2]]
    else:
        # no edges occur in the graph
        edge_index = torch.zeros((0, 2), dtype=torch.int64)
        edge_attr = torch.tensor(bondfeat, dtype=torch.float32)

    # turn into graph object
    data = Data(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr)
    return data



if __name__ == "__main__":
    print(onehot([0,0,0,0,0,1,1,1,1,2,2,2,2,3], 4))