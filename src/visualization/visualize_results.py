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
    experiment_id = "experiment_2"

    # make leaf loss plot
    make_leaf_flow_loss_plot(experiment_id)
    
    make_rewards_plot(experiment_id)

    make_reward_threshold_plot([-1,-2,-3,-4], experiment_id)

    # make rewards 
    smiles = []
    with gzip.open("results/experiment_2/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass

    print(len(smiles))


    """ rewards = []
    with gzip.open("results/experiment_1/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    print(len(rewards))
    rids = np.arange(len(rewards))
    rewards = np.array(rewards)
    rewards_T = lambda T : np.cumsum(rewards > T)
    thresholds = [4,5,6,7,7.5]
    fig, ax = plt.subplots(1,len(thresholds), figsize=(10,3))
    for i, T in enumerate(thresholds):
        ax[i].plot(rids, rewards_T(T))
        ax[i].set_title(f"T={T}")
    plt.show() """
    
    """ k = 1000
    
    top_k = np.sort(rewards[:k])

    top_k_avg = [np.mean(top_k) for _ in range(k+1)]

    for i in range(k+1,len(rewards)):
        if rewards[i] > top_k[0]:
            top_k[0] = rewards[i]
        top_k_avg.append(np.mean(top_k))
        top_k = np.sort(top_k)


    
    plt.semilogx(rids, top_k_avg)
    plt.show() """
    

    
    
    """ best_mol = np.argmax(rewards)


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
                     num_conv_steps=10) """
    
    # get the latest model checkpoint - if none simply start from scratch
    """ param_id = 92000
    
    params = pickle.load(gzip.open(f"models/runs/experiment_1/params_{param_id}.pkl.gz"))
    
    for a,b in zip(model.parameters(), params):
        a.data = torch.tensor(b, dtype=torch.double)

    device = torch.device("cpu")
    for traj in trajs[300000:300001]:
        mol = BlockMolecule()
        for (bi, si) in traj:
            mol.add_block(bi,si)
            batch = [mol.to_block_graph(device=device) for mol in [mol]]

            mols_graph_batch = Batch.from_data_list([graph for graph in batch if graph is not None])
            mols_graph_batch.to(device)
            stem_out, mol_out, _ = model(mols_graph_batch)
            print(stem_out)
            print(mol_out) """

        