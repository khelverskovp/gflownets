import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
from typing import List

# keeps track of the domain of possible blocks
class BlockDictionary:
    def __init__(self, bpath: str) -> None:
        """
        :param bpath: path to json file with smiles block definitions
         
        """

        # load block definitions from file
        blocks = pd.read_json(bpath)

        # load smile names and attachment points
        self.block_smis = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()

        # retrieve the rdkit representation of each molecule
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in self.block_smis]
        self.block_natoms = [m.GetNumAtoms() for m in self.block_mols]

# class that defines a specific molecule
class BlockMolecule:
    def __init__(self, bpath: str) -> None:
        """
        :param bpath: path to json file with smiles block definitions
         
        """

        # block dictionary
        self.bdict = BlockDictionary(bpath)

        # defining characteristics of a molecule
        self.blockidxs = []  # index of the block in the blocks dictionary
        self.blocks = []      # rdkit representation of block
        self.slices = [0]     # start atom index for each block
        self.jbonds = []      # junction bonds: [block1, block2, atomidxblock1, atomidxblock2] 
        self.stems = []       # possible bond attachment points for the molecule
        self.numblocks = 0    # number of blocks in the molecule
    
    def add_block(self, blockidx: int, stemidx: int=0) -> None:
        # add the block to the molecule lists
        self.blockidxs.append(blockidx)
        # get smiles string
        smi = self.bdict.block_smis[blockidx]
        self.blocks.append(Chem.MolFromSmiles(smi))
        self.slices.append(self.slices[-1] + self.bdict.block_natoms[blockidx])

        # update the number of blocks in the molecule
        self.numblocks += 1

        # add all the stem points for the block
        for i, stem in enumerate(self.bdict.block_rs[blockidx]):
            # if it is not the first block added to the molecule 
            # its first stem will be used to connect the block to the rest of the molecule
            # it should therefore not be present in the list of available stems
            if i == 0 and self.numblocks > 1:
                continue
            self.stems.append([self.numblocks-1,stem]) # i.e. the index of the newest added block will be numblocks-1

        # if the molecule is not empty we need to connect the new block to the molecule
        # this is done through a junction bond
        if self.numblocks > 1:
            # by default new blocks are added to the first stem in the stems list for the current molecule
            # i.e. stemidx=0 by default
            block1,stem1 = self.stems[stemidx]
            
            # the newly added block will use its first stem to connect the molecule
            block_r = self.bdict.block_rs[blockidx]
            block2,stem2 = self.numblocks-1, block_r[0]

            # create a junction bond between the stem and the newly added block
            jbond = [block1,block2,stem1,stem2]
            self.jbonds.append(jbond)
            self.stems.pop(stemidx)
    
    def get_smiles(self) -> List[str]:
        return [self.bdict.block_smis[idx] for idx in self.blockidxs]

        

def mol_from_frag(jun_bonds, frags=None, frag_smis=None, coord=None, optimize=False):
    "joins 2 or more fragments into a single molecule"
    jun_bonds = np.asarray(jun_bonds)
    #if jun_bonds.shape[0] == 0: jun_bonds = np.empty([0,4])
    if frags is not None:
        pass
    elif frags is None and frag_smis is not None:
        frags = [Chem.MolFromSmiles(frag_name) for frag_name in frag_smis]
    else:
        raise ValueError("invalid argument either frags or frags smis should be not None")
    if len(frags) == 0: return None, None
    nfrags = len(frags)
    # combine fragments into a single molecule
    mol = frags[0]
    for i in np.arange(nfrags-1)+1:
        mol = Chem.CombineMols(mol, frags[i])
    # add junction bonds between fragments
    frag_startidx = np.concatenate([[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0)[:-1]

    if jun_bonds.size == 0:
        mol_bonds = []
    else:
        mol_bonds = frag_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]

    emol = Chem.EditableMol(mol)

    [emol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE) for bond in mol_bonds]
    mol = emol.GetMol()
    atoms = list(mol.GetAtoms())

    def _pop_H(atom):
        nh = atom.GetNumExplicitHs()
        if nh > 0: atom.SetNumExplicitHs(nh-1)

    [(_pop_H(atoms[bond[0]]), _pop_H(atoms[bond[1]])) for bond in mol_bonds]
    #print([(atom.GetNumImplicitHs(), atom.GetNumExplicitHs(),i) for i,atom in enumerate(mol.GetAtoms())])
    Chem.SanitizeMol(mol)
    # create and optimize 3D structure
    if optimize:
        assert not "h" in set([atm.GetSymbol().lower() for atm in mol.GetAtoms()]), "can't optimize molecule with h"
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.RemoveHs(mol)
    return mol, mol_bonds

    
if __name__ == "__main__":
    bpath = "data/raw/blocks_PDB_105.json"
    molecule = BlockMolecule(bpath=bpath)
    
    for i in range(5):
        molecule.add_block(np.random.randint(0,105))

    
    print(f"blockidxs: {molecule.blockidxs}")
    print(f"slices: {molecule.slices}")
    print(f"jbonds: {molecule.jbonds}")
    print(f"stems: {molecule.stems}")

    mol = mol_from_frag(molecule.jbonds,frag_smis=molecule.get_smiles())[0]
    print(Chem.MolToSmiles(mol))
    mol2 = Chem.MolFromSmiles(molecule.get_smiles()[0])
    print(mol)
    print(mol2)
    _ = AllChem.Compute2DCoords(mol)

    # draw to file
    path = f'reports/figures/molecules/builds/test.png'
    Draw.MolToFile(mol,path)
    





        
    



