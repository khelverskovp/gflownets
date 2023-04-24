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
    experiment_id = "experiment_4"

    #k_values = [10,100,1000]
    #make_top_k_plot(k_values, experiment_id)

    #make_empirical_density_plot()

    #make_diverse_bemis_murcko_plot(7.5, "experiment_1")

    make_leaf_flow_loss_plot(experiment_id)

    #make_tanimoto_plot(experiment_id)
    rewards = []
    path = f'results/{experiment_id}'
    with gzip.open(f"{path}/inflows.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass

    print(rewards)