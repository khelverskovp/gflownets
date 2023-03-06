import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
import numpy as np
from typing import List
from matplotlib.colors import ColorConverter

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
        self.colors = ["red","green","blue","orange","purple","pink","cyan","olive"]
    
    def build_translation_table(self):
        """build a symmetry mapping for blocks. Necessary to compute parent transitions"""
        self.translation_table = {}
        for blockidx in range(len(self.block_mols)):
            # Blocks have multiple ways of being attached. By default,
            # a new block is attached to the target stem by attaching
            # it's kth atom, where k = block_rs[new_block_idx][0].
            # When computing a reverse action (from a parent), we may
            # wish to attach the new block to a different atom. In
            # the blocks library, there are duplicates of the same
            # block but with block_rs[block][0] set to a different
            # atom. Thus, for the reverse action we have to find out
            # which duplicate this corresponds to.

            # Here, we compute, for block blockidx, what is the index
            # of the duplicate block, if someone wants to attach to
            # atom x of the block.
            # So atom_map[x] == bidx, such that block_rs[bidx][0] == x
            atom_map = {}
            for j in range(len(self.block_mols)):
                if self.block_smis[blockidx] == self.block_smis[j]:
                    atom_map[self.block_rs[j][0]] = j
            self.translation_table[blockidx] = atom_map

        # We're still missing some "duplicates", as some might be
        # symmetric versions of each other. For example, block CC with
        # block_rs == [0,1] has no duplicate, because the duplicate
        # with block_rs [1,0] would be a symmetric version (both C
        # atoms are the "same").

        # To test this, let's create nonsense molecules by attaching
        # duplicate blocks to a Gold atom, and testing whether they
        # are the same.
        print(self.translation_table)
        print("")
        gold = Chem.MolFromSmiles('[Au]')
        # If we find that two molecules are the same when attaching
        # them with two different atoms, then that means the atom
        # numbers are symmetries. We can add those to the table.
        for blockidx in range(len(self.block_mols)):
            for j in self.block_rs[blockidx]:
                if j not in self.translation_table[blockidx]:
                    symmetric_duplicate = None
                    for atom, block_duplicate in self.translation_table[blockidx].items():
                        molA, _ = chem.mol_from_frag(
                            jbonds=[[0,1,0,j]],
                            frags=[gold, self.block_mols[blockidx]])
                        molB, _ = chem.mol_from_frag(
                            jbonds=[[0,1,0,atom]],
                            frags=[gold, self.block_mols[blockidx]])
                        if (Chem.MolToSmiles(molA) == Chem.MolToSmiles(molB) or
                            molA.HasSubstructMatch(molB)):
                            symmetric_duplicate = block_duplicate
                            break
                    if symmetric_duplicate is None:
                        raise ValueError('block', blockidx, self.block_smis[blockidx],
                                         'has no duplicate for atom', j,
                                         'in position 0, and no symmetrical correspondance')
                    self.translation_table[blockidx][j] = symmetric_duplicate
                    print(blockidx,j,symmetric_duplicate)
                    #print('block', blockidx, '+ atom', j,
                    #      'in position 0 is a symmetric duplicate of',
                    #      symmetric_duplicate)
        print(self.translation_table)

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
    
    def draw_mol_to_file(self,name: str="test", highlightBonds: bool=False) -> None:
        """
        params:
        name: filename
        """

        # retrieve mol in rdkit format as well as bonds
        mol,bonds = molecule.get_mol_from_jbonds()
        
        # number of bonds
        nbonds = len(bonds)

        # create drawing surface
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)

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
        filename = f'{name}.png'
        d.WriteDrawingText(path + filename)

if __name__ == "__main__":
    bpath = "data/raw/blocks_PDB_105.json"
    molecule = BlockMolecule(bpath=bpath)
    
    """ for i in range(4):
        try:
            molecule.add_block(np.random.randint(0,105))
        except:
            break

    
    print(f"blockidxs: {molecule.blockidxs}")
    print(f"slices: {molecule.slices}")
    print(f"jbonds: {molecule.jbonds}")
    print(f"stems: {molecule.stems}")

    molecule.draw_mol_to_file("marcus",highlightBonds=True) """

    bdict = molecule.bdict

    bdict.build_translation_table()

    






        
    



