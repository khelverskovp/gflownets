import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np


class VisualizerZINC:
    # load data
    filename = "data/processed/docked_mols.csv"

    # set output dir
    out_dir = "reports/figures/molecules/ZINC/"

    def __init__(self):
        self.df = df = pd.read_csv(self.filename)
        self.smiles = self.df["smiles"]

    def visualizeMolecule(self,idx,subdir=""):
        # get smiles representation
        smi = self.smiles[idx]
        # retrieve chemical structure
        m = Chem.MolFromSmiles(smi)
        # compute 2D structure
        _ = AllChem.Compute2DCoords(m)
        # draw to file
        path = f'{self.out_dir}{subdir}mol{idx}.png'
        Draw.MolToFile(m,path)
        # write log
        print("Visualized molecule:")
        print(f"SMILES: {smi}")
        print(f"Path: {path}\n")

    # visualize molecules where the smiles strings where the length is between some prefixed lengths
    def visualizeMolecules(self,idxs=None,min_length=15,max_length=20,n_mols=2,subdir="small/"):
        if idxs == None:
            lengths = np.array([len(s) for s in self.smiles])
            # retrieve the indices of the molecules
            idxs = np.arange(len(self.smiles))[(lengths >= min_length) & (lengths <= max_length)][:n_mols]
        for idx in idxs:
            self.visualizeMolecule(idx,subdir=subdir)


visualizer = VisualizerZINC()  

min_length = 120
max_length = 200
n_mols = 1

#visualizer.visualizeMolecules(min_length=min_length,max_length=max_length,n_mols=n_mols,subdir="big/")
visualizer.visualizeMolecules(idxs=[22, 91, 33, 20, 42],subdir="general/")

m = Chem.MolFromSmiles("O=S=O")
# compute 2D structure
_ = AllChem.Compute2DCoords(m)
# draw to file
path = f'reports/figures/test8.png'

""" for i, atom in enumerate(m.GetAtoms()):
    if i >= 12 and i < 21:
        print(atom.GetAtomicNum()) """

Draw.MolToFile(m,path)

print(visualizer.df.iloc[39])

print(visualizer.df["jbonds"][39])

m = Chem.MolFromSmiles("c1ncc2nc[nH]c2n1")
for atom in m.GetAtoms():
    print(atom.GetAtomicNum())



