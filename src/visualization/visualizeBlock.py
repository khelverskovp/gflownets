import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np
import os


class VisualizerBlock:
    # blocks path
    filename = "data/raw/blocks_PDB_105.json"

    # set output dir
    out_dir = "reports/figures/molecules/blocks/"

    def __init__(self):
        # load block
        blocks = pd.read_json(self.filename)

        # load smile names and attachment points
        self.block_smis = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()

    def visualizeBlock(self,idx,show_stems=False):
        # get smiles representation
        smi = self.block_smis[idx]
        # retrieve chemical structure
        m = Chem.MolFromSmiles(smi)
        # compute 2D structure
        _ = AllChem.Compute2DCoords(m)
        # draw to file
        subdir = "stems/" if show_stems else "no_stems/"
        path = f'{self.out_dir}{subdir}'
        os.makedirs(path,exist_ok=True)
        filename = f'block{idx}.png'
        path = path + filename
        stems = self.block_rs[idx] if show_stems else []
        Draw.MolToFile(m,path,highlightAtoms=stems)
        # write log
        print("Visualized block:")
        print(f"SMILES: {smi}")
        print(f"Path: {path}\n")

if __name__ == "__main__":
    visualizer = VisualizerBlock()  

    for idx in [0,4,10]:
        m = Chem.MolFromSmiles(visualizer.block_smis[idx])
        print(f"Block {idx} has {m.GetNumAtoms()} atoms")
        print(visualizer.block_smis[idx])
        for atom in m.GetAtoms():
            print(atom.GetAtomicNum())


