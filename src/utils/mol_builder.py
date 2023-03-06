import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
import numpy as np
from typing import List
from matplotlib.colors import ColorConverter

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
    
    def get_smiles(self) -> List[str]:
        return [self.bdict.block_smis[idx] for idx in self.blockidxs]


    def get_mol_from_jbonds(self) -> Mol:
        """
        represents the molecule in rdkit.Chem format
        return: molecule in rdkit.Chem.rdchem.Mol form
        """

        # 
        jun_bonds = np.asarray(self.jbonds)
        if len(self.blocks) == 0: return None, None
        
        # combine blocks into a single molecule
        mol = self.blocks[0]
        for i in np.arange(1,self.numblocks):
            mol = Chem.CombineMols(mol, self.blocks[i])
        
        # add junction bonds between fragments
        frag_startidx = np.asarray(self.slices[:-1])

        # check if any junction bonds exists
        if jun_bonds.size == 0:
            mol_bonds = []
        else:
            # creates list of bonds
            # e.g.
            # jun_bonds = [[0,1,0,0],[0,2,1,1]]
            # frag_startidx = [0,6,8]
            # mol_bonds = [[0,6],[1,9]]
            mol_bonds = frag_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]
        
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
    
    for i in range(9):
        try:
            molecule.add_block(np.random.randint(0,105))
        except:
            break

    
    print(f"blockidxs: {molecule.blockidxs}")
    print(f"slices: {molecule.slices}")
    print(f"jbonds: {molecule.jbonds}")
    print(f"stems: {molecule.stems}")

    molecule.draw_mol_to_file("marcus",highlightBonds=True)

    






        
    



