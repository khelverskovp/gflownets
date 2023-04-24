import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import os
from src.utils.mols import BlockMolecule
from src.utils.proxy import Proxy
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles
import time
import pandas as pd

# import defaultdict
from collections import defaultdict

# figure 18 (done)
def make_leaf_flow_loss_plot(experiment_id):
    # path to loss file
    losses_path = f'results/{experiment_id}'

    #To load from pickle file
    leaf_losses = []
    leaf_losses_min = []
    leaf_losses_max = []
    flow_losses = []
    flow_losses_min = []
    flow_losses_max = []

    with gzip.open(f"{losses_path}/losses.pkl.gz") as fr:
        try:
            while True:
                data = pickle.load(fr)
                leaf_losses.extend(data["term_losses"])
                leaf_losses_min.extend(data["term_losses_min"])
                leaf_losses_max.extend(data["term_losses_max"])
                flow_losses.extend(data["flow_losses"])
                flow_losses_min.extend(data["flow_losses_min"])
                flow_losses_max.extend(data["flow_losses_max"])
        except EOFError:
            pass

    steps = []
    lls = []
    llsmin = []
    llsmax = []
    fls = []
    flsmin = []
    flsmax = []
    nbins = 30
    for i in range(6):
        # create bin
        left = 10**i
        right = min(10**(i+1),len(leaf_losses))
        bins = np.linspace(left, right, nbins+1)
        for j in range(nbins):
            steps.append((bins[j]+bins[j+1]) / 2)
            lls.append(np.mean(leaf_losses[int(bins[j]):int(bins[j+1]+1)]))
            llsmin.append(np.min(leaf_losses[int(bins[j]):int(bins[j+1]+1)]))
            llsmax.append(np.max(leaf_losses[int(bins[j]):int(bins[j+1]+1)]))
            fls.append(np.mean(flow_losses[int(bins[j]):int(bins[j+1]+1)]))
            flsmin.append(np.min(flow_losses[int(bins[j]):int(bins[j+1]+1)]))
            flsmax.append(np.max(flow_losses[int(bins[j]):int(bins[j+1]+1)]))

    
    
    plt.figure()
    plt.loglog(steps, lls, color="blue")
    plt.loglog(steps, fls, color="orange")
    plt.fill_between(steps, llsmin, llsmax, color="blue", alpha=0.2)
    plt.fill_between(steps, flsmin, flsmax, color="orange", alpha=0.2)

    plt.ylim(0.00001,6000)
    plt.yticks([0.001,0.01,0.1,1,10,100,1000])

    plt.legend(["leaf loss", "flow loss"])

    plt.xlabel("SGD steps")
    plt.ylabel("loss")
    plt.title("Flow and leaf losses")

    file_id = len(leaf_losses)
    figures_path = f"reports/figures/{experiment_id}/{file_id}"
    os.makedirs(figures_path,exist_ok=True)

    # save file
    filename = f"{figures_path}/leafflowloss_{file_id}.png"
    plt.savefig(filename)

# Rewards plot med moving average (done)
def make_rewards_plot(experiment_id):
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass

    # make a plot of reward on y axis vs molecules generated (length of rewards list) on x axis
    rids = np.arange(len(rewards)) + 1
    rewards = np.array(rewards)

    # plot moving average on top of rewards
    rewards_ma = np.convolve(rewards, np.ones((100,))/100, mode='valid')
    rids_ma = np.arange(len(rewards_ma))

    plt.figure()
    plt.semilogx(rids, rewards)
    plt.semilogx(rids_ma, rewards_ma)
    plt.xlabel("Molecules generated")
    plt.ylabel("Reward")
    plt.title("Reward")

    #add legends
    plt.legend(["Reward", "Moving average"])

    

    file_id = len(rids)
    figures_path = f"reports/figures/{experiment_id}/{int(len(rids) / 4)}"
    os.makedirs(figures_path,exist_ok=True)


    filename = f"{figures_path}/rewards_{file_id}.png"
    plt.savefig(filename)

    
# figure 5 (without bemis murcko with 4 thresholds)
def make_reward_threshold_plot(thresholds, experiment_id):
    # only allow for 4 thresholds
    assert len(thresholds) == 4, "Please give 4 thresholds for plot to work!"
    # path to rewards file
    rewards_path = f'results/{experiment_id}'

    # store rewards in list
    rewards = []

    with gzip.open(f"{rewards_path}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass

    # change to numpy array
    rewards = np.array(rewards)
    rids = np.arange(len(rewards)) + 1

    # compute number of molecules with rewards > T
    rewards_T = lambda T : np.cumsum(rewards > T)
    
    # create figure
    fig, axs = plt.subplots(2,2)
    
    for ax, T in zip(axs.ravel(), thresholds):
        ax.plot(rids, rewards_T(T))
        ax.set_xlabel("states visited")
        ax.set_ylabel(f"# of modes with R>{T}")
        ax.grid(True)
    
    figures_path = f"reports/figures/{experiment_id}/{int(len(rids) / 4)}"
    os.makedirs(figures_path,exist_ok=True)

    fig.suptitle(f"Number of high-reward molecules with different thresholds")
    fig.tight_layout()

    # save file
    file_id = len(rewards)
    filename = f"{figures_path}/reward_threshold_plot_{file_id}.png"
    plt.savefig(filename)

# figure 4 (not unique molecules)
def make_top_k_plot(k_values, experiment_id):
    # only allow for 3 thresholds
    assert len(k_values) == 3, "Please give 3 thresholds for plot to work!"
    plt.figure()
    # path to rewards file is experiment_id/unique_rewards.pkl.gz
    path = f'results/{experiment_id}'

    # store rewards in list
    rewards = []

    with gzip.open(f"{path}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    # store smiles in a list 
    smiles = []
    with gzip.open(f"{path}/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass
    
    is_sampled = defaultdict(lambda:False)

    # change to numpy array
    rewards = np.array(rewards)
    linestyles = [':','--','-']
    legends = ["top 10", "top 100", "top 1000"]

    for k, ls, lg in zip(k_values, linestyles, legends):
        top_k = np.sort(rewards[:k])

        top_k_avg = [np.mean(top_k)]

        for i in range(k+1,len(rewards)):
            if rewards[i] > top_k[0] and not is_sampled[smiles[i]]:
                top_k[0] = rewards[i]
                is_sampled[smiles[i]] = True
            top_k_avg.append(np.mean(top_k))
            top_k = np.sort(top_k)

        top_k_rids = np.arange(k,len(rewards))
        plt.semilogx(top_k_rids, top_k_avg, ls, label=lg)
        
    plt.ylim(2,8.5)
    plt.yticks([2,4,6,8])
    plt.minorticks_off()
    plt.grid()
    plt.xlabel("molecules visited")
    plt.ylabel("avg " + r"$R$" + " of top " + r"$k$")

    plt.legend()
    file_id = len(rewards)
    figures_path = f"reports/figures/{experiment_id}/{int(file_id / 4)}"
    os.makedirs(figures_path,exist_ok=True)
    
    # save file
    filename = f"{figures_path}/top_k_reward_plot_{file_id}.png"
    plt.savefig(filename)

# figure 3 (empirical density of rewards)
# for beta = 1, beta = 4 and beta = 10 and proxy dataset
def make_empirical_density_plot(experiment_id, betas):
    # load data points
    filename = "docked_mols.csv"
    path = f"data/processed/{filename}"

    df = pd.read_csv(path)

    # get rewards for dataset
    df_rewards = df["dockscore"].values

    
    #get rewards for experiment
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass

    # change to numpy array
    rewards = np.array(rewards)

    linestyles = [':','--','-']

    # for each beta in betas add a legend \beta$ = value of beta 

   # legends = [f"\beta$={beta}" for beta in betas]
    legends = [r'$\beta$ = 10', r'$\beta$ = 1', r'$\beta$ = 4', 'proxy dataset']

    # and add "proxy dataset to the legend"
    legends.append("proxy dataset")

    pdf_proxy, bins_proxy = np.histogram(df_rewards, density=True)
    pdf_proxy = pdf_proxy / np.sum(pdf_proxy)

    # plot pdf_proxy and pdf for each beta. The x axis is 0-8 (R(x)) and the y axis is the probability density
    
    plt.xlim(0,8)
    #make xlim 0, 2, 4, 6, 8
    plt.xticks([0,2,4,6,8])

    # plot pdf_proxy
    plt.plot(bins_proxy[:-1], pdf_proxy, label=legends[-1])

    # plot pdf for each beta
    for beta, ls, lg in zip(betas, linestyles, legends[:-1]):
        # get pdf for beta
        pdf_beta, bins_beta = np.histogram(rewards, density=True)
        pdf_beta = pdf_beta / np.sum(pdf_beta)

        # plot pdf for beta
        plt.plot(bins_beta[:-1], pdf_beta, ls, label=lg)
    

    plt.xlabel("reward")
    plt.ylabel("empirical density")
    plt.title("Empirical density of rewards",fontsize=14)
    plt.legend()
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)

    # save file
    file_id = len(rewards)
    filename = f"{figures_path}/empirical_density_plot_{file_id}.png"
    plt.savefig(filename)




def make_empirical_density_proxy():
    # load data points
    filename = "docked_mols.csv"
    path = f"data/processed/{filename}"

    df = pd.read_csv(path)

    # get rewards for dataset
    df_rewards = df["dockscore"].values

    pdf_proxy, bins_proxy = np.histogram(df_rewards, density=True)
    pdf_proxy = pdf_proxy / np.sum(pdf_proxy)

    # plot pdf_proxy
    plt.figure()
    plt.plot(bins_proxy[:-1], pdf_proxy, label="proxy dataset")
    plt.xlabel("reward")
    plt.ylabel("empirical density")

    plt.title("Empirical density of rewards",fontsize=14)
    plt.legend()
    figures_path = f"reports/figures/proxy"
    os.makedirs(figures_path,exist_ok=True)

    # save file
    file_id = len(df_rewards)
    filename = f"{figures_path}/empirical_density_plot_proxy_{file_id}.png"
    plt.savefig(filename)

def make_scaffold_plot(experiment_id):
    #get rewards for experiment
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    smiles = []
    with gzip.open(f"results/{experiment_id}/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass
    
    smiles = smiles[:10000]
    rewards = rewards[:10000]
        
    from collections import defaultdict
    start_time = time.time()
    unique_scaffold = defaultdict(lambda: True)
    T = 7.5
    is_bemis_murcko = np.zeros(len(rewards))
    for i, smi in enumerate(smiles):
        bemis_murcko = MurckoScaffoldSmiles(smi)
        if rewards[i] > T and unique_scaffold[bemis_murcko]:
            is_bemis_murcko[i] = 1
            unique_scaffold[bemis_murcko] = False
    plt.plot(np.arange(len(smiles)),np.cumsum(is_bemis_murcko))
    plt.show()

def make_tanimoto_plot(experiment_id):
    #get rewards for experiment
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    smiles = []
    with gzip.open(f"results/{experiment_id}/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass
    
    smiles = smiles[:10000]
    rewards = rewards[:10000]
    
    T = 7.5 


    print(count)
        
    


if __name__ == "__main__":
    experiment_id = "experiment_1"
    # path to rewards file
    rewards_path = f'results/{experiment_id}'
    trajs = []
    with gzip.open(f"{rewards_path}/trajectories.pkl.gz") as fr:
        try:
            while True:
                trajs.extend(pickle.load(fr))
        except EOFError:
            pass
    
    trajs = trajs
    begin_time = time.time()
    smiles = []
    tl = {key: 0 for key in range(2,9)}
    count = np.zeros(len(trajs)+1)
    for i, traj in enumerate(trajs):
        """ if i % 100 == 0:
            print(i)
        mol = BlockMolecule()
        for (bi, si) in traj:
            mol.add_block(bi,si)

        smiles.append(mol.get_smiles())
        if smiles[-1] == "CC(O)C1CCCC1":
            print(traj) """
        tl[len(traj)] += 1
        count[i+1] = traj[-1][0] == -1
    
    
    print(np.cumsum(count))
    plt.plot(np.arange(len(count)),np.cumsum(count))
    plt.show()
    
    print("Blocksize for generated molecules:", tl)
    
    #print(time.time()-begin_time)
    """ d = {key: 0 for key in np.unique(smiles)}
    
    for i in range(len(smiles)):
        d[smiles[i]] += 1
    
    tuples = [(d[key],key) for key in np.unique(smiles)]
    tuples.sort()
    sorted = list(reversed(tuples))
    #print(sorted) """

    smiles = []
    with gzip.open(f"{rewards_path}/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass
    
    print(len(np.unique(smiles[:4000])))
    
    


    
    