import pickle
import torch
import gzip
import time
import numpy as np
from src.utils.mols import BlockMolecule
from src.utils.proxy import Proxy
from src.models.model import GFlownet
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import os
import pandas as pd

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == "__main__":
    #To load from pickle file
    leaf_losses = []
    leaf_losses_min = []
    leaf_losses_max = []
    flow_losses = []
    flow_losses_min = []
    flow_losses_max = []
    with gzip.open("results/experiment_1/losses.pkl.gz") as fr:
        try:
            while True:
                data = pickle.load(fr)
                leaf_losses.extend(data["leaf_losses"])
                leaf_losses_min.extend(data["leaf_losses_min"])
                leaf_losses_max.extend(data["leaf_losses_max"])
                flow_losses.extend(data["flow_losses"])
                flow_losses_min.extend(data["flow_losses_min"])
                flow_losses_max.extend(data["flow_losses_max"])
        except EOFError:
            pass
    
    print(len(flow_losses))
    steps = np.arange(len(leaf_losses))

    plt.figure()
    plt.loglog(steps, leaf_losses, color="blue")
    plt.loglog(steps, flow_losses, color="orange")
    plt.fill_between(steps, leaf_losses_min, leaf_losses_max, color="blue", alpha=0.1)
    plt.fill_between(steps, flow_losses_min, flow_losses_max, color="orange", alpha=0.1)

    plt.ylim(0.0001,2000)
    plt.yticks([0.0001,0.001,0.01,0.1,1,10,100,1000])

    plt.legend(["leaf loss", "flow loss"])

    plt.xlabel("SGD steps")
    plt.ylabel("loss")
    plt.title("Flow and leaf losses")

    path = f'reports/figures/experiment_1'
    os.makedirs(path,exist_ok=True)

    filename = f"{path}/leafflowloss_92000.png"
    plt.savefig(filename)
    plt.show()
    
    
    #print(data_losses)

    """ rewards = []
    with gzip.open("results/experiment_1/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    print(len(rewards))
    rewards = np.array(rewards)
    best_mol = np.argmax(rewards)

    print(f"From results: {rewards[best_mol]}")

    print(np.sum(rewards > 7.5))
    print(len(rewards))
    print(np.mean(rewards))

    molgen = np.arange(len(rewards))
    rewardt = np.cumsum(rewards > 7.5)

    plt.plot(molgen,rewardt)
    plt.show()

    plt.hist(rewards)
    plt.show()

    pdf, bins = np.histogram(rewards, density=True)
    pdf = pdf / np.sum(pdf)

    # Plot the empirical PDF
    plt.plot(bins[:-1], pdf, 'b', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Empirical PDF')
    plt.show()

    trajs = []
    with gzip.open("results/experiment_1/trajectories.pkl.gz") as fr:
        try:
            while True:
                trajs.extend(pickle.load(fr))
        except EOFError:
            pass
    
    
    proxy = Proxy(device=torch.device("cpu"))
    mol = BlockMolecule()
    for (bi, si) in trajs[best_mol]:
        mol.add_block(bi,si)
    mol.draw_mol_to_file("worst_mol",highlightBonds=True)
    print(f"From proxy using trajectory: {proxy([mol]).item()}") """
    
    """ trajs = []
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

        print(model(mols_graph_batch)) """

        