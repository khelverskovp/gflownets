import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    def visualizeBlock(self,idx,figsize=(300,300),show_stems=False):
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
        stems = [self.block_rs[idx][0]] if show_stems else []
        Draw.MolToFile(m,path,size=figsize,highlightAtoms=stems)
        # write log
        print("Visualized block:")
        print(f"SMILES: {smi}")
        print(f"Path: {path}\n")
    
    def visualizeBlockWithDefaultStem(self,idx,figsize=(300,300)):
        # get smiles representation
        smi = self.block_smis[idx]
        # retrieve chemical structure
        m = Chem.MolFromSmiles(smi)
        # compute 2D structure
        _ = AllChem.Compute2DCoords(m)
        # draw to file
        subdir = "default_stems/"
        path = f'{self.out_dir}{subdir}'
        os.makedirs(path,exist_ok=True)
        filename = f'block{idx}.png'
        path = path + filename
        stems = [self.block_rs[idx][0]]
        Draw.MolToFile(m,path,size=figsize,highlightAtoms=stems)
        # write log
        print("Visualized block:")
        print(f"SMILES: {smi}")
        print(f"Path: {path}\n")

    def getAllBlocksWithDefaultStemPlot(self):
        fig, ax = plt.subplots(nrows=7,ncols=15,figsize=(15,7))

        for idx in range(len(self.block_smis)):
            self.visualizeBlockWithDefaultStem(idx,figsize=(300,150))

            path = f'{self.out_dir}default_stems/block{idx}.png'
            img = mpimg.imread(path)
        
            ax[idx // 15, idx % 15].imshow(img)
            ax[idx // 15, idx % 15].axis('off')
        
        fig.suptitle("SMILES block visualization with default stems",fontweight='bold')
        plt.savefig("reports/figures/allstems.png")


if __name__ == "__main__":
    visualizer = VisualizerBlock() 

    #visualizer.getAllBlocksWithDefaultStemPlot()

    print(visualizer.block_smis[12])
    print(visualizer.block_smis[23])
    

    



