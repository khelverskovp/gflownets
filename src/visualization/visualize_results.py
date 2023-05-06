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
from collections import defaultdict

if __name__ == "__main__":
    #make_empirical_density_plot()

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
    
    get_number_of_diverse_bemis_murcko(7.5, "experiment_1")
    get_number_of_diverse_bemis_murcko(7.5, "experiment_4")
    get_number_of_diverse_bemis_murcko(7.5, "experiment_5")

    #make_leaf_flow_loss_plot(experiment_id)

