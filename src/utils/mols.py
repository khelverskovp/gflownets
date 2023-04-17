import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import numpy as np
from typing import List
from matplotlib.colors import ColorConverter
import os

import src.utils.chem as chem
from torch_geometric.data import Data, Batch
import torch

from src.utils.chem import atomic_numbers

# keeps track of the domain of possible blocks
# cache to store library for features such as acceptor and donor characteristics
_mpnn_feat_cache = [None]

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
        self.colors = ["yellow","lightgreen","orange","purple","pink","cyan","olive","red"]

        # define the set of unique blocks and the length of the set
        self.unique_block_set = sorted(set(self.block_smis))
        self.n_unique_blocks = len(self.unique_block_set)

        # define indexes to represent available stems for each unique block
        self.stem_type_offset = np.int32([0] + list(np.cumsum([max(self.block_rs[self.block_smis.index(smi)])+1 for smi in self.unique_block_set])))
        # the last index represents no stem. Used for empty molecules and molecules with no available stems
        self.n_stem_types = self.stem_type_offset[-1]

        # define a list to map blockidx (0-104) to unique idx (0-71) because of duplicates
        self.true_blockidx = [self.unique_block_set.index(smi) for smi in self.block_smis]


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
                    symmetric_duplicate = None
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
                            symmetric_duplicate = block
                            break
                        
                    if symmetric_duplicate is None:
                        # if we could not find a symmetric duplicate we raise an error. 
                        raise ValueError("Could not find symmetric duplicate for block {} and stem {}".format(block_i, j))
                    else:
                        # else we add the symmetric duplicate to the translation table
                        self.translation_table[block_i][j] = symmetric_duplicate

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
        self.atom_nfeatures = 6 + 8 + len(chem.atomic_numbers)
    
    @property
    def stem_atmidxs(self):
        stems = np.asarray(self.stems)
        if stems.shape[0]==0:
            stem_atmidxs = np.array([])
        else:
            stem_atmidxs = np.asarray(self.slices)[stems[:,0]] + stems[:,1]
        return stem_atmidxs
    
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
    
    def get_smiles(self) -> List[str]:
        """
        returns smiles string for the molecule
        """
        return Chem.MolToSmiles(self.mol_to_rdkit())


    def mol_to_rdkit(self, return_bonds=False) -> Mol:
        """
        represents the molecule in rdkit.Chem format
        return: molecule in rdkit.Chem.rdchem.Mol form
        """
        if return_bonds:
            return chem.mol_from_jbonds_and_blocks(self.jbonds, blocks=self.blocks)

        return chem.mol_from_jbonds_and_blocks(self.jbonds, blocks=self.blocks)[0]
    
    def draw_mol_to_file(self,name: str="test", highlightBonds: bool=False, figsize=(500,500)) -> None:
        """
        params:
        name: filename
        """

        # retrieve mol in rdkit format as well as bonds
        mol,bonds = self.mol_to_rdkit(return_bonds=True)
        
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
    
    # code is taken from https://github.com/GFNOrg/gflownet/blob/master/mols/utils/chem.py
    # comments are written by us
    def mpnn_feat(self):
        """
            returns atom and bond features in encoded format for the mpnn network

            the feature vector will have size (natoms, 70)

            for each atom in the molecule a one hot encoded vector of size 70 exists
            index 0: 1 if atom is H otherwise 0
            index 1: 1 if atom is C otherwise 0
            index 2: 1 if atom is N otherwise 0
            index 3: 1 if atom is O otherwise 0
            index 4: 1 if atom is F otherwise 0
            index 5: 1 if index 0-4 is 0 otherwise 1
            index 7: 1 if atom is acceptor otherwise 0
            index 8: 1 if atom is donor otherwise 0
            index 9: 1 if the atom is part of an aromatic ring otherwise 0
            index 10: 1 if atom is part of sp hybridization otherwise 0
            index 11: 1 if atom is part of sp2 hybridization otherwise 0
            index 12: 1 if atom is part of sp3 hybridzation otherwise 0
            index 13: number of hydrogen atoms connected to the atom
            index 14-69: onehot encoded atom vector describing which atom is present
            e.g. if index 15 is 1 it means that the atom is helium
        """

        # encode which main atoms and bonds are present
        atomtypes = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.UNSPECIFIED: 0}

        # convert to rdkit format
        rdmol = self.mol_to_rdkit()

        # count number of atoms in the molecule
        natm = len(rdmol.GetAtoms())
        ntypes = len(atomtypes)

        # number of features that appear in the encoded vector for each atom
        nfeat = ntypes + 1 + 8 + len(atomic_numbers)
        
        # make embedding for each atom
        atmfeat = np.zeros((natm, nfeat))

        # featurize
        for i, atom in enumerate(rdmol.GetAtoms()):
            # check if the atom is in atomtypes otherwise set to 5
            type_idx = atomtypes.get(atom.GetSymbol(), 5)
            # set the corresponding index to 1
            atmfeat[i, type_idx] = 1

            # set index 9 to 1 if atom is aromatic
            atmfeat[i, ntypes + 4] = atom.GetIsAromatic()

            # onehot encode hybridization type accordingly (see description in top)
            hybridization = atom.GetHybridization()
            atmfeat[i, ntypes + 5] = hybridization == HybridizationType.SP
            atmfeat[i, ntypes + 6] = hybridization == HybridizationType.SP2
            atmfeat[i, ntypes + 7] = hybridization == HybridizationType.SP3

            # set index 13 to number of hydrogen atoms attached
            atmfeat[i, ntypes + 8] = atom.GetTotalNumHs(includeNeighbors=True)

            # set the corresponding atom index in the one hot encoded vector to 1
            # e.g. H has atomic num 1, which means that
            # atmfeat[i,14] = 1 if the ith atom is hydrogen 
            atmfeat[i, ntypes + 9 + atom.GetAtomicNum() - 1] = 1

        # get donor and acceptor information
        if _mpnn_feat_cache[0] is None:
            # load feature library from rdkit if not stored in cache
            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            _mpnn_feat_cache[0] = factory
        else:
            factory = _mpnn_feat_cache[0]

        # load features for the molecule
        feats = factory.GetFeaturesForMol(rdmol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Acceptor':
                # set index 7 for all acceptor atoms to 1
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atmfeat[k, ntypes + 2] = 1
            elif feats[j].GetFamily() == 'Donor':
                # set index 8 for all donor atoms to 1
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atmfeat[k, ntypes + 3] = 1
            
        # get bonds and bond features
        bond = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in rdmol.GetBonds()])
        # a list representing the type of each bond
        # 0 -> single bond
        # 1 -> double bond
        # 2 -> triple bond
        # 3 -> aromatic bond

        bondfeat = [bondtypes[bond.GetBondType()] for bond in rdmol.GetBonds()]

        # onehot encodes bond features
        bondfeat = chem.onehot(bondfeat, num_classes=len(bondtypes) - 1)
        
        return atmfeat, bond, bondfeat

    def to_atom_graph(self, floatX=torch.float):
        # if the molecule is empty simply return an empty graph of zeroes
        if len(self.blockidxs) == 0:
            # Data class is used to represent as a graph
            g = Data(x=torch.zeros((1, 14 + len(atomic_numbers))), 
                     edge_attr=torch.zeros((0, 4)), 
                     edge_index=torch.zeros((0, 2)).long())
        else:
            # atmfeat contains a onehot encoded vector for each atom in the molecule
            # bond represents the starting and end atom indices for each junction bond
            # bondfeat contains a onehot encoded vector for each bond in the molecule
            # e.g. if bondfeat[0] = [0,1,0,0] it means that the first bond uses a double bond
            atmfeat, bond, bondfeat = self.mpnn_feat()

            # convert to torch geometric format
            g = chem.mol_to_graph_backend(atmfeat, bond, bondfeat)

        # get atom indices for atoms in self.stems
        atoms_in_open_stems = self.stem_atmidxs
        if not len(atoms_in_open_stems):
            # if no open stems appear set the first value to 1 at index 70
            atoms_in_open_stems = [0]

        # create stem mask which is initially 0 for all atoms
        stem_mask = torch.zeros((g.x.shape[0], 1))
        # becomes an additional feature to the onehot encoded atom vector at index 70
        # 1 if the atom is an open stem otherwise 0
        stem_mask[torch.tensor(atoms_in_open_stems).long()] = 1

        # add to g.x
        g.x = torch.cat([g.x, stem_mask], 1).to(floatX)
        
        g.edge_attr = g.edge_attr.to(floatX)
        if g.edge_index.shape[0] == 0:
            g.edge_index = torch.zeros((2, 1)).long()
            g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).to(floatX)
        return g
    
    def to_block_graph(self,device):
        # define lambda function to convert lists to torch tensors 
        f = lambda x: torch.tensor(x, dtype=torch.long, device=device)

        # an empty molecule is represented as block 72 with stemtype 214
        if len(self.blockidxs) == 0:
            g = Data(
                x=f([self.bdict.n_unique_blocks]),
                edge_index=f([[],[]]),
                edge_attr=f(([],[])).T,
                stems=f([(0,0)]),
                stemtypes=f([self.bdict.n_stem_types]))
            g.to(device)
            return g
        
        # else the molecule is not empty
        # edges are equal to (block1,block2) for each bond in jbonds. List that maps which blocks have edges.
        edges = [(jbond[0],jbond[1]) for jbond in self.jbonds]

        # unique blocks and stem_type_offset are used to convert the blockidxs to the correct block type
        true_blocks = self.bdict.true_blockidx 
        stem_offset = self.bdict.stem_type_offset

        # get unique stemidx for each bond. 
        # each block has a unique range of values that describes values for stems
        # e.g. stems on "C1=CNC=CC1" are defined by values in the range 4-6. i.e. a value of 5 means that atom 1 on "C1=CNC=CC1" is used in the given junction bond
        edge_attrs = [(stem_offset[true_blocks[self.blockidxs[jbond[0]]]] + jbond[2],
                       stem_offset[true_blocks[self.blockidxs[jbond[1]]]] + jbond[3])
                      for jbond in self.jbonds]
    
        # get unique stem idx for open stems
        stemtypes = [stem_offset[true_blocks[self.blockidxs[i[0]]]] + i[1] for i in self.stems]

        # create the graph
        g = Data(
            x=f([true_blocks[i] for i in self.blockidxs]),
            edge_index=f(edges).T if len(edges) > 0 else f([[],[]]),
            edge_attr=f(edge_attrs) if len(edges) > 0 else f([]).reshape((0,2)),
            stems=f(self.stems) if len(self.stems) > 0 else f([(0,0)]),
            stemtypes=f(stemtypes) if len(self.stems) > 0 else f([self.bdict.n_stem_types]))
        
        g.to(device) # moves the data to the device (cpu or gpu)
        return g


class MoleculeMDP:
    def __init__(self):
        self.molecule = BlockMolecule()
        # initialize translation table
        self.translation_table = self.molecule.bdict.get_translation_table()
    
    def parents(self, mol):
        """
            return a list of parent states
            [(parent_molecule in BlockMolecule format, (blockidx, stemidx)),...]
            blockidx: index of block that should be added to the parent to get the current state
            stemidx: stemidx in parent_molecule.stems that the block should connect with

            inspired by code from the github of Bengio
        """


        # if the molecule exist of the only one block there exist only one parent state
        if mol.numblocks == 1:
            return [(BlockMolecule(), (mol.blockidxs[0],0))]

        # compute the degree for each block in the molecule
        # initialize to zero
        degree = {i: 0 for i in range(mol.numblocks)}
        for (block1, block2, _, _) in mol.jbonds:
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
            parent_mol = mol.copy()

            # delete the block from parent_mol
            removed_stem = parent_mol.delete_block_with_degree_one(ridx)

            # get block index from the deleted block which is still stored in the original self.molecule
            blockidx = mol.blockidxs[ridx]

            # when the block was deleted from the molecule the stem it was attached to was placed in the end of parent_mol.stems
            stemidx = len(parent_mol.stems) - 1

            # define parent
            parent = [parent_mol, (self.translation_table[blockidx][removed_stem[1]], stemidx)]

            # add to list of parents
            parent_mols.append(parent)
            

        return parent_mols
    
    def mol_to_batch(self,mols):
        batch = Batch.from_data_list(mols)
        batch.to(self.device)
        return batch

    
    
            

if __name__ == "__main__":
    molecule = BlockMolecule()

    # build molecule
    # choose molecules to add
    """ block_list = [91, 29, 48, 95]

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

    print(f"to block graph: {molecule.to_block_graph()}")
    print(f"to block graph x: {molecule.to_block_graph().x}")
    print(f"to block graph edge_index: {molecule.to_block_graph().edge_index}")
    print(f"to block graph edge_attr: {molecule.to_block_graph().edge_attr}")
    print(f"to block graph stems: {molecule.to_block_graph().stems}")
    print(f"to block graph stemtypes: {molecule.to_block_graph().stemtypes}") """

    """ max_atomic_num = 0

    for block in range(105):
        molecule = BlockMolecule()
        molecule.add_block(block)
        graph = molecule.to_atom_graph()
        for atom in graph.x:
            if atom[66].item():
                print(block)
                molecule.draw_mol_to_file("test50") """
    
    molecule = BlockMolecule()
    molecule.add_block(0)
    molecule.add_block(5)
    molecule.add_block(6)
    graph = molecule.to_block_graph()

    print(graph.x)
    print(graph.edge_index)
    print(graph.edge_attr)
    print(graph.stems)
    print(graph.stemtypes)

    print(sorted(set(molecule.bdict.block_smis)))
    # debug
    #import pdb
    #pdb.set_trace()
    

    
    




    






        
    



