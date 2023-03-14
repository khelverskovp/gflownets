# inspired by code from chem.py of the initial paper repository


import numpy as np
from rdkit import Chem


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
