import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
import numpy as np
from typing import List
from matplotlib.colors import ColorConverter
import os

import chem

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

        # define colors for molecule plots
        # i.e. highlighting colors of bonds
        # self.colors = ["red","green","blue","orange","purple","pink","cyan","olive"]
        self.colors = ["yellow","lightgreen"]

    def build_translation_table(self):
        """
        build translation table for symmetrical attachments to molecules.
        used to compute parent states
        inspired by code from E. Bengio github repo on gflownets
        """

        # create dictionary of dictionaries
        # a dictionary for each blockidx
        # i.e. self.translation_table[blockidx]: dict
        # each dict match from a stem to a blockidx
        # i.e. if self.translation_table[5][0] = 4
        # it means that using stem 0 on block 5 is exactly the same as attaching block 4 with its first stem

        self.translation_table = {}

        # since symmetry is obtained by duplicating blocks and rearranging the stems in block_r
        # e.g. block 4, 5 and 6 all have the same smiles string "C1CCNCC1"
        # however 
        # self.block_rs[4] = [0,3]
        # self.block_rs[5] = [1,0,3]
        # self.block_rs[6] = [3,0]
        # by default when a new block is added to the molecule it is attached using stem self.block_rs[blockidx][0]

        # the first entries in the translation table then be found using the following logic
        # for each block_i from 0 to 104
        # create an empty dict atom_map
        # for each block_j with the same smiles string as block_i
        # set atom_map[self.block_rs[block_j][0]] = block_j
        # lastly set self.translation_table

        # e.g. when block_i = 4
        # smi1 == smi2, when block_j in {4,5,6}
        # when block_j = 4
        # atom_map[self.block_rs[4][0]]=atom_map[0]=4
        # when block_j = 5
        # atom_map[self.block_rs[5][0]]=atom_map[1]=5
        # when block_j = 6
        # atom_map[self.block_rs[6][0]]=atom_map[3]=6

        # i.e. self.translation_table[4] = {0: 4, 1: 5, 3: 6}
        # using stem 0 on block 4 is the same as using block 4 with the default stem
        # using stem 1 on block 4 is the same as using block 5 with the default stem
        # using stem 3 on block 4 is the same as using block 6 with the default stem

        for block_i in range(len(self.block_smis)):
            atom_map = {}
            for block_j in range(len(self.block_smis)):
                smi1, smi2 = self.block_smis[block_i], self.block_smis[block_j]

                if smi1 == smi2:
                    atom_map[self.block_rs[block_j][0]] = block_j

            self.translation_table[block_i] = atom_map
        
        print(self.translation_table)

        # some duplicates will still be missing 
        # e.g CC has block_r = [0,1]
        # however no duplicate exist in the blocks list since attaching to either makes
        # no difference symmetrically

        # in order to discover these duplicates we build on top on some base molecule
        mol = Chem.MolFromSmiles("Ir")
        for block_i in range(len(self.block_mols)):
            # loop over all possible stems in the block
            stem1 = self.block_rs[block_i][0]
            for block_j in range(len(self.block_mols)):
                # get default stem for second block
                stem2 = self.block_rs[block_j][0]

                # check that the block does not already exist in the translation table
                if stem2 not in self.build_translation_table[block_i].keys():
                    # create junction bond for the first molecule
                    jbond1 = [0,1,0,stem1]
                    mol1,_ = chem.mol_from_frag(jbonds=[jbond1],frags=[mol,self.block_mols[block_i]])

                    # create junction bond for the second molecule
                    jbond2= [0,1,0,stem2]
                    mol2,_ = chem.mol_from_frag(jbonds=[jbond2],frags=[mol,self.block_mols[block_j]])

                    # check if the to molecules are identical
                    if Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2) or mol1.HasSubtructMatch(mol2):
                        pass
                        



        

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
        # ensure that we can actually add a block to the molecule
        assert (self.numblocks == 0 or len(self.stems) > 0), "No open stems! Cannot add block to molecule" 

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

    def remove_jbond(self,jbond_idx = None) -> None:
        """"""
        pass
    
    def get_smiles(self) -> List[str]:
        """
        returns list of smiles string for the blocks in the molecule
        """
        return [self.bdict.block_smis[idx] for idx in self.blockidxs]


    def get_mol_from_jbonds(self) -> Mol:
        """
        represents the molecule in rdkit.Chem format
        return: molecule in rdkit.Chem.rdchem.Mol form
        """

        return chem.mol_from_frag(self.jbonds, frags=self.blocks)
    
    def draw_mol_to_file(self,name: str="test", highlightBonds: bool=False, figsize=(500,500)) -> None:
        """
        params:
        name: filename
        """

        # retrieve mol in rdkit format as well as bonds
        mol,bonds = molecule.get_mol_from_jbonds()
        
        # number of bonds
        nbonds = len(bonds)

        # create drawing surface
        h, w = figsize
        d = rdMolDraw2D.MolDraw2DCairo(h, w)

        # check highlighting condition

        if highlightBonds:
            # get colors from block dictionary and convert to tuple format
            colors = [ColorConverter().to_rgb(color) for color in self.bdict.colors][:nbonds]

            # list to store highlighted atoms and bonds
            highlightedAtoms = []
            highlightedBonds = []

            # dictionaries to store atom and bond colors
            atom_cols = {}
            bond_cols = {}

            for bond, color in zip(bonds,colors):
                # get atom indices
                # convert to integers since rdkit has problems with numpy.int64
                atm1 = int(bond[0])
                atm2 = int(bond[1])

                # add atoms to highlighted atoms list
                highlightedAtoms.extend([atm1,atm2])

                # update colors
                atom_cols[atm1] = color
                atom_cols[atm2] = color

                # get bond index
                bond_idx = mol.GetBondBetweenAtoms(atm1,atm2).GetIdx() 

                # add bond to highlighted bonds
                highlightedBonds.append(bond_idx)

                # update color
                bond_cols[bond_idx] = color
          
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightedAtoms,
                                           highlightAtomColors=atom_cols,
                                           highlightBonds=highlightedBonds,
                                           highlightBondColors=bond_cols)
        else:
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        
        # save image to file
        path = f'reports/figures/molecules/builds/'
        os.makedirs(path,exist_ok=True)
        filename = f'{name}.png'
        d.WriteDrawingText(path + filename)

if __name__ == "__main__":
    bpath = "data/raw/blocks_PDB_105.json"
    molecule = BlockMolecule(bpath=bpath)
    
    # build molecule
    # choose molecules to add
    block_list = [12]

    for block in block_list:
        try:
            molecule.add_block(block)
        except:
            break

    
    print(f"blockidxs: {molecule.blockidxs}")
    print(f"slices: {molecule.slices}")
    print(f"jbonds: {molecule.jbonds}")
    print(f"stems: {molecule.stems}")

    filename = "NewMolecule3"
    molecule.draw_mol_to_file(filename,highlightBonds=True, figsize=(500,250))

    # bdict = molecule.bdict

    # bdict.build_translation_table()

    






        
    



