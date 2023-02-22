# import rdkit
from __future__ import print_function 
import rdkit
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# load some molecules from docked_mols.csv
df = pd.read_csv("data/processed/docked_mols.csv")

# get index 2 and 480 of df 
df = df.iloc[[5,195],:]

# visualize the molecules 
for i in range(len(df)):
    mol = Chem.MolFromSmiles(df.iloc[i,0])
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300,300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open("reports/figures/molecules/mol_%s.svg" % i, "w") as f:
        f.write(svg)
    print("Molecule %s saved" % i)


