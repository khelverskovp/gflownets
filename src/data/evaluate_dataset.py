import pickle
import torch
import gzip
import time
import numpy as np
from src.utils.mols import BlockMolecule
from src.utils.proxy import Proxy
from src.utils.plots import *
from src.models.model import GFlownet
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

# load data points
filename = "docked_mols.csv"
path = f"data/processed/{filename}"

df = pd.read_csv(path)

columns = df.columns 

for name in columns[2:]:
    df.loc[:,name] = df[name].apply(json.loads)

proxy = Proxy(device=torch.device("cpu"))

rewards = []

begin_time = time.time()

use_proxy = False

# loop over the number of rows
for i in range(len(df)):
    if use_proxy:
        # get the row
        row = df.iloc[i]
        # get the stem
        stems = row["stems"]
        # get jbonds
        jbonds = row["jbonds"]
        # get blockidx
        blockidxs = row["blockidxs"]
        # get slices
        slices = row["slices"]
        # get smiles
        smiles = row["smiles"]

        # make the molecule
        mol = BlockMolecule()
        # set the stem
        mol.stems = stems
        # set the jbonds
        mol.jbonds = jbonds
        # set the blockidx
        mol.blockidxs = blockidxs
        # set the slices
        mol.slices = slices
        
        # set the blocks
        #loop over the blockidxs
        for blockidx in mol.blockidxs:
            smi = mol.bdict.block_smis[blockidx]
            mol.blocks.append(Chem.MolFromSmiles(smi))
        
        # append proxy([mol]).item() to rewards
        rewards.append(proxy([mol]).item())
    else:
        mu, std = [-8.6, 1.1]
        transform = lambda v: 4 - (v-mu) / std
        rewards.append(transform(df.dockscore[i]))

    end_time = time.time()
    # for every 100 print end_time - begin_time
    if i % 10000 == 0:
        print(f"{i} / {len(df)}")
        print(f"Time: {end_time - begin_time}")
    
    
    


# save the rewards
ext = "_true" if not use_proxy else ""
pickle.dump(rewards, gzip.open(f"data/processed/rewards_proxy_dataset{ext}.pkl.gz", "wb"))