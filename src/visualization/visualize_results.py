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
    experiment_id = "experiment_3"
    d = defaultdict(lambda: False)
    i = 0
    with gzip.open(f"results/{experiment_id}_base/trajectories.pkl.gz") as fr:
        try:
            while True:
                data = pickle.load(fr)
                if i != 233: 
                    pickle.dump(data,
                        gzip.open(f"results/{experiment_id}/trajectories.pkl.gz", 'ab'))
                i += 1        
        except EOFError:
            pass
    
    
    k_values = [10,100,1000]
    #make_top_k_plot(k_values, experiment_id)

    T = 7.5
    #make_diverse_bemis_murcko_plot(T, experiment_id)

    #make_leaf_flow_loss_plot(experiment_id)
