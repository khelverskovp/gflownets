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
    experiment_id = "experiment_2"
    k_values = [10,100,1000]
    #make_top_k_plot(k_values, experiment_id)

    T = 7
    S = 0.7
    #make_tanimoto_plot(T, S, experiment_id)

    T = 8
    #make_diverse_bemis_murcko_plot(T, experiment_id)

    #make_scatter_inflow_reward_plot(experiment_id)

    make_empirical_density_inflow_reward_plot(experiment_id)



    #make_leaf_flow_loss_plot(experiment_id)

