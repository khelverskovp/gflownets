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

path = "results/experiment_1/"

smiles = []
with gzip.open(f"{path}smiles.pkl.gz") as fr:
    try:
        while True:
            smiles.extend(pickle.load(fr))
    except EOFError:
        pass

rewards = []
with gzip.open(f"{path}rewards.pkl.gz") as fr:
    try:
        while True:
            rewards.extend(pickle.load(fr))
    except EOFError:
        pass


    
smiles_reward = {}
for i in range(len(smiles)):
    smiles_reward[smiles[i]] = rewards[i]



# get the rewards from the dictionary in a list
unique_rewards = list(smiles_reward.values())

print(len(unique_rewards))

# save the rewards from the dictionary to a file called unique_rewards
with gzip.open(f"{os.getcwd()}/data/processed/unique_rewards.pkl.gz", "wb") as fw:
    pickle.dump(unique_rewards, fw)

