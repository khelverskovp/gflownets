import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np

#print(rdkit.__version__)

""" m = Chem.MolFromSmiles("Cc1ccccc1")

print(m.GetNumAtoms())

for atom in m.GetAtoms():
    print(atom.GetAtomicNum()) """


in_dir = "reports/figures/molecules/"

# load data points
filename = "data/processed/docked_mols.csv"
df = pd.read_csv(filename)

#print(df.tail()["jbonds"])


pairs = []


""" for i in range(len(df["blockidxs"])):
    if i % 1000 == 0:
        print(i)
    for j in range(i+1,len(df["blockidxs"])):
        if i != j:
            if len(df["blockidxs"][i]) == len(df["blockidxs"][j]):
                if np.all(np.array(df["blockidxs"][i]) == np.array(df["blockidxs"][j])):
                    pairs.append((i,j))
                    print(i,j) """


ms = [Chem.MolFromSmiles(smi) for smi in df["smiles"][:2]]

#ms = [Chem.MolFromSmiles(smi) for smi in ["O=c1nccc[nH]1"]]

temp = sorted([(len(df["smiles"][i]), i) for i in range(len(df["smiles"]))])

print(temp[10])

print(df["smiles"][temp[10][1]])

for m in ms: tmp=AllChem.Compute2DCoords(m)

for i, m in enumerate(ms): Draw.MolToFile(m,f'{in_dir}mol{i}.png')    


