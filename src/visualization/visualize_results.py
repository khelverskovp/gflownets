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
    experiment_id = "experiment_1"

    k_values = [10,100,1000]
    make_top_k_plot(k_values, experiment_id)

    