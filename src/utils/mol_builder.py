import pandas as pd
from rdkit import Chem

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
        self.block_smis = [] # smiles name of each block in the molecul
        self.block_idxs = [] # index of the block in the self.blocks dictionary
        self.slices = [0] # start atom index for each block
        self.jbonds = [] # junction bonds: [block1, block2, atomidxblock1, atomidxblock2] 
        self.stems = [] # possible bond attachment points for the molecule
        self.numblocks = 0 # number of blocks in the molecule
    
    def add_block(self, blockidx: int):
        # add the block to the molecule lists
        self.block_smis.append(self.bdict.block_smis[blockidx])
        self.block_idxs.append(blockidx)
        self.slices.append(self.slices[-1] + self.bdict.block_natoms[blockidx])

        # update the number of blocks in the molecule
        self.numblocks += 1

        # add all the stem points for the block
        for stem in self.bdict.block_rs[blockidx]:
            self.stems.append([self.numblocks-1,stem]) # i.e. the index of the newest added block will be numblocks-1

        # if the molecule is empty simply add the block information
        if self.numblocks == 0:
            pass
            




        
        




if __name__ == "__main__":
    # path to block definitions
    blocks_file = "data/raw/blocks_PDB_105.json"

    # initialize an empty molecule
    mol = BlockMolecule(blocks_file)

    print(mol.bdict.block_natoms)



