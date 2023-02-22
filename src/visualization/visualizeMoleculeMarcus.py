import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

#print(rdkit.__version__)

""" m = Chem.MolFromSmiles("Cc1ccccc1")

print(m.GetNumAtoms())

for atom in m.GetAtoms():
    print(atom.GetAtomicNum()) """


in_dir = "reports/figures/molecules/"

template = Chem.MolFromSmiles('c1nccc2n1ccc2')

ms = [Chem.MolFromSmiles(smi) for smi in ('OCCc1ccn2cnccc12','C1CC1Oc1cc2ccncn2c1','CNC(=O)c1nccc2cccn12')]

for m in ms: tmp=AllChem.Compute2DCoords(m)
from rdkit.Chem import Draw
Draw.MolToFile(ms[0],f'{in_dir}cdk2_mol1.o.png')    
Draw.MolToFile(ms[1],f'{in_dir}cdk2_mol2.o.png')
Draw.MolToFile(ms[2],f'{in_dir}cdk2_mol3.o.png')

