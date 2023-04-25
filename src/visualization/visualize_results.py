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

if __name__ == "__main__":
    # experiment_id
    """ experiment_id = "experiment_1"

    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    print(np.max(rewards[:52000]),np.mean(rewards[:52000]))

    experiment_id = "experiment_4"

    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    print(np.max(rewards),np.mean(rewards))
    """
    experiment_id = "experiment_2"
    
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    print(np.max(rewards),np.mean(rewards))
    print(len(rewards))

    experiment_id = "experiment_2"
    k_values = [10,100,1000]
    make_top_k_plot(k_values, experiment_id)

    make_empirical_density_plot()
    make_leaf_flow_loss_plot(experiment_id)

    #make_diverse_bemis_murcko_plot(7.5, "experiment_1")

    #make_tanimoto_plot(experiment_id)

    #experiment_id = ["experiment_4","experiment_5"]
    #[make_empirical_density_inflow_reward_plot(eid) for eid in experiment_id]
    #make_empirical_density_inflow_reward_plot("experiment_1")
    #make_scatter_inflow_reward_plot("experiment_2")