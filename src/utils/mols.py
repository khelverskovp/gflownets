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
from torch_geometric.data import Data
import torch

# keeps track of the domain of possible blocks
class BlockDictionary:
    # path to json file with smiles block definitions
    bpath = "data/raw/blocks_PDB_105.json"

    def __init__(self) -> None:

        # load block definitions from file
        blocks = pd.read_json(self.bpath)

        # load smile names and attachment points
        self.block_smis = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()

        # retrieve the rdkit representation of each molecule
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in self.block_smis]
        self.block_natoms = [m.GetNumAtoms() for m in self.block_mols]

        # define colors for molecule plots
        # i.e. highlighting colors of bonds
        # self.colors = ["red","green","blue","orange","purple","pink","cyan","olive"]
        # self.colors = ["yellow","lightgreen"]
        self.colors = ["yellow","lightgreen","orange"]

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

        # some duplicates will still be missing 
        # e.g CC has block_r = [0,1]
        # however no duplicate exist in the blocks list since attaching to either makes
        # no difference symmetrically

        # now we find symmetric duplicates by starting from a random molecule Ir and continually building two molecles mol1 and mol2
        # if mol1 and mol2 are the same then we have found a symmetric duplicate and we add it to the translation table
        base_mol = Chem.MolFromSmiles("[Ir]")
        # we loop over all blocks
        for block_i in range(len(self.block_smis)):
            # loop over all stems of the block
            for j in self.block_rs[block_i]:
                # if the stem is already in the translation table we dont need to do anything
                if j not in self.translation_table[block_i].keys():
                    symmetric_dubplicate = None
                    # loop over stems and blockidx in the translation table for the block
                    for stem, block in self.translation_table[block_i].items():
                        # blocks is a list of the base molecule and the block we are checking in rdkit format
                        blocks = [base_mol, self.block_mols[block_i]]
                        # create mol1 by attaching block_i to base_mol using stem j
                        # jbonds1 is [[0,1,0,j]] is the bond between mol and block_i using stem j
                        jbonds1 = [[0,1,0,j]]
                        mol1,_ = chem.mol_from_jbonds_and_blocks(jbonds1, blocks)  
                        # create mol2 by attaching block_i to mol using stem stem
                        # jbonds2 is [[0,1,0,stem]] is the bond between mol and block_i using stem, stem
                        jbonds2 = [[0,1,0,stem]]
                        mol2,_ = chem.mol_from_jbonds_and_blocks(jbonds2, blocks)

                        # now we check if mol1 and mol2 are symmetric dubplicates 
                        if Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2) or mol1.HasSubstructMatch(mol2):
                            # if they are we add the block to the translation table
                            symmetric_dubplicate = block
                            break
                        
                    if symmetric_dubplicate is None:
                        # if we could not find a symmetric duplicate we raise an error. 
                        raise ValueError("Could not find symmetric duplicate for block {} and stem {}".format(block_i, j))
                    else:
                        # else we add the symmetric duplicate to the translation table
                        self.translation_table[block_i][j] = symmetric_dubplicate

    def get_translation_table(self):
        self.build_translation_table()
        return self.translation_table

# class that defines a specific molecule
class BlockMolecule:
    def __init__(self) -> None:
        # block dictionary
        self.bdict = BlockDictionary()

        # defining characteristics of a molecule
        self.blockidxs = []  # index of the block in the blocks dictionary
        self.blocks = []      # rdkit representation of block
        self.slices = [0]     # start atom index for each block
        self.jbonds = []      # junction bonds: [block1, block2, atomidxblock1, atomidxblock2] 
        self.stems = []       # possible bond attachment points for the molecule
        self.numblocks = 0    # number of blocks in the molecule

        # number of features in atomic graph representation
        # first 6 elements are used for typeidx, next 8 elements are for other features (e.g hybridization type),
        # remaining 56 elements represents an onehot encoded version of the molecule with the first 56 atoms in the periodic table
        # 6 + 8 + 56 = 70
        self.atom_nfeatures = 70
    
    # make copy of existing molecule
    def copy(self):
        new_mol = BlockMolecule()
        new_mol.blockidxs = self.blockidxs.copy()
        new_mol.blocks = self.blocks.copy()
        new_mol.slices = self.slices.copy()
        new_mol.jbonds = self.jbonds.copy()
        new_mol.stems = self.stems.copy()
        new_mol.numblocks = self.numblocks
        return new_mol

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
        returns smiles string for the molecule
        """
        return Chem.MolToSmiles(self.mol_to_rdkit()[0])


    def mol_to_rdkit(self) -> Mol:
        """
        represents the molecule in rdkit.Chem format
        return: molecule in rdkit.Chem.rdchem.Mol form
        """
        return chem.mol_from_jbonds_and_blocks(self.jbonds, blocks=self.blocks)
    
    def draw_mol_to_file(self,name: str="test", highlightBonds: bool=False, figsize=(500,500)) -> None:
        """
        params:
        name: filename
        """

        # retrieve mol in rdkit format as well as bonds
        mol,bonds = self.mol_to_rdkit()
        
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
    
    def delete_block_with_degree_one(self,blockidx: int) -> None:
        """
        deletes blocks from the molecule
        """
        mask = np.ones(self.numblocks,dtype=bool)
        mask[blockidx] = 0

        # update blockidxs
        self.blockidxs = list(np.array(self.blockidxs)[mask])

        # update blocks
        self.blocks = list(np.array(self.blocks)[mask])

        # update numblocks
        self.numblocks = self.numblocks - 1

        # update slices
        nr_atoms = [self.bdict.block_natoms[bidx] for bidx in self.blockidxs]
        self.slices = [0] + list(np.cumsum(nr_atoms))

        # update jbonds
        reindex = np.cumsum(mask) - 1
        jbonds = []
        r_stem = []
        free_stem = []
        for (block1, block2, stem1, stem2) in self.jbonds:
            if block1 == blockidx:
                r_stem = [block1, stem1]
                free_stem = [reindex[block2], stem2]
                continue
            elif block2 == blockidx:
                r_stem = [block2, stem2]
                free_stem = [reindex[block1], stem1]
                continue
            jbonds.append([reindex[block1], reindex[block2], stem1, stem2])
        self.jbonds = jbonds

        # update stems
        stems = []
        for (block, stem) in self.stems:
            if block == blockidx:
                continue
            stems.append([reindex[block], stem])
        self.stems = stems
        self.stems.append(free_stem)

        return r_stem
    
    def to_atom_graph(self):
        #check if the molecule is not empty
        if len(self.blockidxs) == 0:
            # g is a graph with no nodes and no edges in torch geometric format (Data)
            g = Data(x=torch.zeros((1, self.atom_nfeatures)),
                 edge_attr=torch.zeros((4,0)),
                 edge_index=torch.zeros((2, 0)).long())

        return g




class MoleculeMDP:
    def __init__(self):
        self.molecule = BlockMolecule()
        # initialize translation table
        self.translation_table = self.molecule.bdict.get_translation_table()

    def add_block(self, blockidx: int, stemidx: int=0) -> None:
        self.molecule.add_block(blockidx=blockidx,stemidx=stemidx)

    
    def parents(self):
        """
            return a list of parent states
            [(parent_molecule in BlockMolecule format, (blockidx, stemidx)),...]
            blockidx: index of block that should be added to the parent to get the current state
            stemidx: stemidx in parent_molecule.stems that the block should connect with

            inspired by code from the github of Bengio
        """


        # if the molecule exist of the only one block there exist only one parent state
        if self.molecule.numblocks == 1:
            return [(BlockMolecule(), (self.molecule.blockidxs[0],0))]

        # compute the degree for each block in the molecule
        # initialize to zero
        degree = {i: 0 for i in range(self.molecule.numblocks)}
        for (block1, block2, _, _) in self.molecule.jbonds:
            degree[block1] += 1
            degree[block2] += 1
        
        # if the degree is larger than 1 the block can not have been added in the previous step
        # only look at blocks with degree 1
        removed_blocks = [idx for idx, deg in degree.items() if deg == 1] 
        
        # store parents in list
        parent_mols = []

        # loop over all the blocks we can infer parent states from
        for ridx in removed_blocks:
            # create a new instance of the molecule
            new_mol = self.molecule.copy()

            # delete the block from new_mol
            removed_stem = new_mol.delete_block_with_degree_one(ridx)

            # get block index from the deleted block which is still stored in the original self.molecule
            blockidx = self.molecule.blockidxs[ridx]

            # when the block was deleted from the molecule the stem it was attached to was placed in the end of new_mol.stems
            stemidx = len(new_mol.stems) - 1

            # define parent
            parent = [new_mol, (self.translation_table[blockidx][removed_stem[1]], stemidx)]

            # add to list of parents
            parent_mols.append(parent)
            

        return parent_mols
    

    
    
            

if __name__ == "__main__":
    molecule = BlockMolecule()

    print(molecule.mol_to_atom_graph().x)
    print(molecule.mol_to_atom_graph().edge_index.shape)
    print(molecule.mol_to_atom_graph().edge_attr.shape)
    print(molecule.mol_to_atom_graph().edge_index)
    
    """ # build molecule
    # choose molecules to add
    block_list = [91, 29, 48, 95]

    for block in block_list:
        try:
            molecule.add_block(block)
        except:
            break

    molecule.jbonds = [[0, 1, 3, 0], [0, 2, 6, 4], [2, 3, 0, 3]]

    print(f"smiles: {molecule.get_smiles()}")
    print(f"blockidxs: {molecule.blockidxs}")
    print(f"slices: {molecule.slices}")
    print(f"jbonds: {molecule.jbonds}")
    print(f"stems: {molecule.stems}")

    #molecule.delete_block_with_degree_one(1)

    #print(f"smiles: {molecule.get_smiles()}")
    #print(f"blockidxs: {molecule.blockidxs}")
    #print(f"slices: {molecule.slices}")
    #print(f"jbonds: {molecule.jbonds}")
    #print(f"stems: {molecule.stems}")

    # filename = "ZINC4_bonds"
    # molecule.draw_mol_to_file(filename,highlightBonds=True, figsize=(500,250))

    #bdict = molecule.bdict
    
    # bdict.build_translation_table()

    mdp = MoleculeMDP()
    print(mdp.translation_table)

    mdp.molecule = molecule

    print("")

    print(mdp.parents())

    print("")
    
    mdp.molecule.draw_mol_to_file(name="test1",highlightBonds=True)

    c = 2

    for mol, (blockidx, stemidx) in mdp.parents():
        print("Parent")
        print(f"smiles: {mol.get_smiles()}")
        print(f"blockidxs: {mol.blockidxs}")
        print(f"slices: {mol.slices}")
        print(f"jbonds: {mol.jbonds}")
        print(f"stems: {mol.stems}")
        print("")
        mol.draw_mol_to_file(name=f"parent{c}",highlightBonds=True)
        c += 1
        org_mol = mol.copy()
        org_mol.add_block(blockidx=blockidx,stemidx=stemidx)
        print("Original")
        print(f"smiles: {org_mol.get_smiles()}")
        print(f"blockidxs: {org_mol.blockidxs}")
        print(f"slices: {org_mol.slices}")
        print(f"jbonds: {org_mol.jbonds}")
        print(f"stems: {org_mol.stems}")
        print("")
        org_mol.draw_mol_to_file(name=f"test{c}",highlightBonds=True)
        c += 1

 """
    # sanity check

    # mdp.molecule = BlockMolecule()

    # mdp.add_block(29)

    # print(mdp.parents())



    






        
    



