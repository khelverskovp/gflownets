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
    # experiment_id
    experiment_id = "experiment_1"
    experiment_ids = ["experiment_1","experiment_4","experiment_5"]

    #make_top_k_plot([10,100,1000], experiment_ids)
    #make_empirical_density_plot()
    #make_bemis_murcko_avg_plot(7)

    #make_bemis_murcko_avg_plot()
    #make_leaf_flow_loss_plot(experiment_id)
    #make_empirical_density_inflow_reward_plot(experiment_id)

    #make_tanimoto_plot(experiment_ids,7,default=True)
    #make_tanimoto_plot(experiment_ids,7)
    #make_total_unique_molecules_plot(experiment_ids,7)

    #thresholds = [5,6,7,7.5]
    #make_blocksize_bar_plot(thresholds,experiment_ids)

    #make_blocksize_distribution_plot()
    #make_nonhydrogen_distribution_plot()
    #make_covariance_matrix_plot()

    #make_proxy_true_density_plot()
