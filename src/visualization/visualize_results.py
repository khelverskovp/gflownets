import pickle
import torch
import gzip
import time
import numpy as np
from src.utils.mols import BlockMolecule
from src.models.model import GFlownet
from torch_geometric.data import Batch

if __name__ == "__main__":
    #To load from pickle file
    data_losses = []
    with gzip.open("results/experiment_1/losses.pkl.gz") as fr:
        try:
            while True:
                data_losses.append(pickle.load(fr)["hp"])
        except EOFError:
            pass

    
    print(data_losses)

    data_mols = []
    with gzip.open("results/experiment_1/rewards.pkl.gz") as fr:
        try:
            while True:
                data_mols.extend(pickle.load(fr))
        except EOFError:
            pass

    trajs = []
    with gzip.open("results/experiment_1/trajectories.pkl.gz") as fr:
        try:
            while True:
                trajs.extend(pickle.load(fr))
        except EOFError:
            pass
    
    model = GFlownet(nemb=256,
                     out_per_stem=105,
                     out_per_stop=1,
                     num_conv_steps=10)
    
    # get the latest model checkpoint - if none simply start from scratch
    param_id = 1
    
    params = pickle.load(gzip.open(f"models/runs/experiment_1/params_{param_id}.pkl.gz"))
    
    for a,b in zip(model.parameters(), params):
        a.data = torch.tensor(b, dtype=torch.double)

    device = torch.device("cpu")
    for traj in trajs:
        mol = BlockMolecule()
        for (bi, si) in traj:
            mol.add_block(bi,si)
        batch = [mol.to_block_graph(device=device) for mol in [mol]]

        mols_graph_batch = Batch.from_data_list([graph for graph in batch if graph is not None])
        mols_graph_batch.to(device)

        print(model(mols_graph_batch))

        