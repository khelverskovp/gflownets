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

rewards_exp1 = []
rewards_exp4 = []
rewards_exp5 = []

smiles_exp1 = []
smiles_exp4 = []
smiles_exp5 = []

with gzip.open(f"results/experiment_1/unique_rewards.pkl.gz") as fr:
    try:
        while True:
            rewards_exp1.extend(pickle.load(fr))
    except EOFError:
        pass

with gzip.open(f"results/experiment_1/smiles.pkl.gz") as fr:
    try:
        while True:
            smiles_exp1.extend(pickle.load(fr))
    except EOFError:
        pass

with gzip.open(f"results/experiment_4/unique_rewards.pkl.gz") as fr:
    try:
        while True:
            rewards_exp4.extend(pickle.load(fr))
    except EOFError:
        pass

with gzip.open(f"results/experiment_4/smiles.pkl.gz") as fr:
    try:
        while True:
            smiles_exp4.extend(pickle.load(fr))
    except EOFError:
        pass


with gzip.open(f"results/experiment_5/unique_rewards.pkl.gz") as fr:
    try:
        while True:
            rewards_exp5.extend(pickle.load(fr))
    except EOFError:
        pass

with gzip.open(f"results/experiment_5/smiles.pkl.gz") as fr:
    try:
        while True:
            smiles_exp5.extend(pickle.load(fr))
    except EOFError:
        pass

# get rewards for proxy dataset data/processed/rewards_proxy_dataset.pkl.gz
df_rewards = pd.read_pickle("data/processed/rewards_proxy_dataset.pkl.gz")
# change to numpy array
df_rewards = np.array(df_rewards)

# get the 10^5 first rewards from each experiment 
rewards_exp1_1 = rewards_exp1[:100000]
rewards_exp4_1 = rewards_exp4[:100000]
rewards_exp5_1 = rewards_exp5[:100000]

def top_k(rewards1, rewards2, rewards3, k):
    mean1 = np.mean(sorted(rewards1, reverse=True)[:k])
    mean2 = np.mean(sorted(rewards2, reverse=True)[:k])
    mean3 = np.mean(sorted(rewards3, reverse=True)[:k])

    mean = np.mean([mean1, mean2, mean3])
    std = np.std([mean1, mean2, mean3])

    return mean, std

# for each rewards_exp1, rewards_exp4, rewards_exp5, get the number of rewards above 8
def get_number_rewards_above_8(rewards):
    rewards_above_8 = []
    for i in range(len(rewards)):
        if rewards[i] >= 8:
            # check if the reward is in the proxy dataset
            if rewards[i] not in df_rewards:
                rewards_above_8.append(rewards[i])
    return rewards_above_8

rewards_exp1_above_8 = get_number_rewards_above_8(rewards_exp1)
rewards_exp4_above_8 = get_number_rewards_above_8(rewards_exp4)
rewards_exp5_above_8 = get_number_rewards_above_8(rewards_exp5)

# print the length of each with a message 
print(f"Number of rewards above 8 for experiment 1: {len(rewards_exp1_above_8)}")
print(f"Number of rewards above 8 for experiment 4: {len(rewards_exp4_above_8)}")
print(f"Number of rewards above 8 for experiment 5: {len(rewards_exp5_above_8)}")

