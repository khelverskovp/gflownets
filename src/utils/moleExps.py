
#import chem
from rdkit import Chem

# create a dictionary of every atom in the periodic table and its atomic number
atom_indexes = {}

for i in range(1, 119): 
    atom_indexes[Chem.GetPeriodicTable().GetElementSymbol(i)] = i

import json

# import blocks_PDB_105.json as a dictionary
with open('blocks_PDB_105.json') as f:
    blocks_PDB_105 = json.load(f)

# create a dictionary from the json file
for key in blocks_PDB_105:
    blocks_PDB_105[key] = blocks_PDB_105[key]['features']

# create a molecule class
class Molecule:
    def __init__(self):
        self.smiles = ''
        self.dockscore = 0
        self.blockidxs = []
        self.slices = []
        self.jbonds = []
        self.stems = []

    # create a function to add a block from blocks_PDB_105 to the molecule
    def add_block(self, blockidx):
        self.blockidxs.append(blockidx)
        if self.smiles == '':
            self.smiles = blocks_PDB_105[blockidx]['block_smi']
        else:
            self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.smiles) + Chem.MolFromSmiles(blocks_PDB_105[blockidx]['block_smi']))
        
        # add the atom index at which the block starts to slices
        self.slices.append(len(self.smiles) - len(blocks_PDB_105[blockidx]['block_smi']))

        # add [block1, block2, bond1, bond2] to jbonds
        

    
