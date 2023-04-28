import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import os
from src.utils.mols import BlockMolecule
from src.utils.proxy import Proxy
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles
import time
import pandas as pd
from scipy import stats

from collections import defaultdict

# Figure 3 
def make_empirical_density_plot():
    # get rewards for experiment for each beta
    rewards_beta1 = []
    rewards_beta4 = []

    with gzip.open(f"results/experiment_2/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards_beta1.extend(pickle.load(fr))
        except EOFError:
            pass
    with gzip.open(f"results/experiment_3/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards_beta4.extend(pickle.load(fr))
        except EOFError:
            pass

    # change to numpy array
    rewards_beta1 = np.array(rewards_beta1)
    rewards_beta4 = np.array(rewards_beta4)

    # get rewards for proxy dataset data/processed/rewards_proxy_dataset.pkl.gz
    df_rewards = pd.read_pickle("data/processed/rewards_proxy_dataset.pkl.gz")
    # change to numpy array
    df_rewards = np.array(df_rewards)

    linestyles = [':', '-', '-']
    colors = ['blue', 'blue', 'black']
    legends = [r'$\hat{p}(R | \beta=1)$', r'$\hat{p}(R | \beta=4)$', r'$\hat{p}(R | \mathrm{proxy\ dataset})$']

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot empirical density for each beta
    for i, rewards in enumerate([rewards_beta1, rewards_beta4, df_rewards]):
        pdf, bins = np.histogram(rewards, density=False, bins=40)
        pdf = pdf / np.sum(pdf)
        ax.plot(bins[:-1], pdf, linestyle=linestyles[i], color=colors[i], label=legends[i])
        

    ax.grid()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.00)

    ax.set_xlabel(r'$R(x)$')
    ax.set_xticks(np.arange(0, 9, 2))
    ax.set_ylabel(r'$\hat{p}(R)$')
    ax.legend()

    figures_path = f"reports/figures"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/empirical_density_plot.png"
    plt.savefig(filename)


# Figure 4
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
    plt.ylabel("avg " + r"$R$" + " of unique top " + r"$k$")

    plt.legend()
    file_id = len(rewards)
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)
    
    # save file
    filename = f"{figures_path}/top_k_reward_plot_{file_id}.png"
    plt.savefig(filename)



# Figure 14

avg = []

def is_diverse_tanimoto(smi, diverse_modes, S):
    fps = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
    for smi_ref in diverse_modes:
        start_time = time.time()
        fps_ref = Chem.RDKFingerprint(Chem.MolFromSmiles(smi_ref))
        tanimoto_sim = DataStructs.FingerprintSimilarity(fps,fps_ref)
        avg.append(time.time()-start_time)
        if tanimoto_sim >= S:
            return False
    return True

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
    
    
    rewards = np.array(rewards)
    smiles = np.array(smiles)
    
    start_time = time.time()
    T = 7 
    S = 0.7

    _,idx = np.unique(smiles[rewards > T], return_index=True)
    smiles = smiles[np.sort(idx)]
    
    diverse_modes = set()
    diverse_tanimoto = np.zeros(len(rewards))
    print(len(smiles))
    for i, smi in enumerate(smiles):
        if i % 100 == 0:
            end_time = time.time()
            print(i, len(diverse_modes), "The last 100 iterations took:", end_time - start_time, "seconds")
            start_time = end_time
            print(np.mean(avg))
        #if is_explored[smi]:
        #    continue
        if is_diverse_tanimoto(smi, diverse_modes, S):
            diverse_modes.add(smi)
            diverse_tanimoto[i] = 1
        #is_explored[smi] = True
    plt.xlim(0,1e6)
    plt.plot(np.arange(len(diverse_tanimoto))+1, np.cumsum(diverse_tanimoto))
    plt.show()


def make_diverse_bemis_murcko_plot(T, experiment_id):
    # get rewards and smiles for experiment 
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

    unique_scaffold = defaultdict(lambda: True)
    is_bemis_murcko = np.zeros(len(rewards))
    start_time = time.time() # get start time
    elapsed_time = 0
    for i, smi in enumerate(smiles):
        bemis_murcko = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi)))
        if rewards[i] > T and unique_scaffold[bemis_murcko]:
            unique_scaffold[bemis_murcko] = False
            is_bemis_murcko[i] = 1
        if i % 10000 == 0:
            end_time = time.time() # get end time
            elapsed_time += end_time - start_time # calculate elapsed time
            print("Iteration ", i, " elapsed time: ", elapsed_time, " seconds")
            start_time = end_time

    # make plot with # of modes with R > T on the ylabel and states visisted on the xlabel
    plt.plot(np.arange(len(smiles)),np.cumsum(is_bemis_murcko))
    plt.xlabel("states visited")
    # set ylabel with the actual T value
    plt.ylabel(f"# of modes with R > {T}")
    plt.grid()

    plt.xticks([0, 0.2*10**6, 0.4*10**6, 0.6*10**6, 0.8*10**6, 1.0*10**6])
    try:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    except:
        import pdb
        pdb.set_trace()

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/diverse_bemis_murcko_plot_{int(len(rewards))}_{T}.png"
    plt.savefig(filename)


# Figure 16
def make_scatter_inflow_reward_plot(experiment_id):
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    inflows = []
    with gzip.open(f"results/{experiment_id}/inflows.pkl.gz") as fr:
        try:
            while True:
                inflows.extend(pickle.load(fr))
        except EOFError:
            pass

    rewards = (np.array(rewards)[-10000:] / 8)**10
    inflows = np.exp(np.array(inflows)[-10000:]) -2.5e-5

    plt.scatter(rewards,inflows)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

# Figure 17
def make_empirical_density_inflow_reward_plot(experiment_id):
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    """ inflows = []
    with gzip.open(f"results/{experiment_id}/inflows.pkl.gz") as fr:
        try:
            while True:
                inflows.extend(pickle.load(fr))
        except EOFError:
            pass """

    rewards = np.array(rewards)[-10000:]
    print(np.max(rewards))
    print(np.sum(rewards))
    rewards /= np.sum(rewards)
    linestyles = ['-', '--']
    colors = ['blue', 'black']
    legends = ["rewards","inflows"] 

    # plot empirical density for each beta
    for i, vals in enumerate([rewards,rewards]):
        pdf, bins = np.histogram(vals, density=False, bins=40)
        pdf = pdf / np.sum(pdf)
        plt.semilogx(bins[:-1], pdf, linestyle=linestyles[i], color=colors[i], label=legends[i])
        

    plt.grid()
    
    plt.xlabel(r'$R(x)$')
    plt.ylabel(r'$\hat{p}(R)$')
    plt.legend()
    plt.xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
    plt.xlim(1e-6,3e-1)

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/empirical_density_inflow_reward.png"
    plt.savefig(filename)


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
    hp = None

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
                hp = data["hp"]
        except EOFError:
            pass

    steps = []
    lls = []
    llsmin = []
    llsmax = []
    fls = []
    flsmin = []
    flsmax = []
    lambda_T = hp["lambda_T"]
    nbins = 20
    
    for i in range(6):
        # create bin
        left = 10**i
        right = min(10**(i+1),len(leaf_losses))
        bins = np.linspace(left, right, nbins+1)
        for j in range(nbins):
            steps.append((bins[j]+bins[j+1]) / 2)
            lls.append(np.mean(leaf_losses[int(bins[j]):int(bins[j+1]+1)]) / lambda_T)
            llsmin.append(np.min(leaf_losses[int(bins[j]):int(bins[j+1]+1)]) / lambda_T)
            llsmax.append(np.max(leaf_losses[int(bins[j]):int(bins[j+1]+1)]) / lambda_T)
            fls.append(np.mean(flow_losses[int(bins[j]):int(bins[j+1]+1)]))
            flsmin.append(np.min(flow_losses[int(bins[j]):int(bins[j+1]+1)]))
            flsmax.append(np.max(flow_losses[int(bins[j]):int(bins[j+1]+1)]))

    
    
    plt.figure()
    plt.loglog(steps, lls, color="blue")
    plt.loglog(steps, fls, color="orange")
    plt.fill_between(steps, llsmin, llsmax, color="blue", alpha=0.2)
    plt.fill_between(steps, flsmin, flsmax, color="orange", alpha=0.2)

    plt.ylim(0.0001,6000)
    plt.yticks([0.001,0.01,0.1,1,10,100,1000])

    plt.legend(["leaf loss", "flow loss"])

    plt.xlabel("SGD steps")
    plt.ylabel("loss")
    plt.grid()
    file_id = len(leaf_losses)
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)

    # save file
    filename = f"{figures_path}/leafflowloss_{file_id}.png"
    plt.savefig(filename)









# Extra plot
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
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)


    filename = f"{figures_path}/rewards_{file_id}.png"
    plt.savefig(filename)

# Extra plot (without bemis murcko with 4 thresholds)
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
    
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)

    fig.suptitle(f"Number of high-reward molecules with different thresholds")
    fig.tight_layout()

    # save file
    file_id = len(rewards)
    filename = f"{figures_path}/reward_threshold_plot_{file_id}.png"
    plt.savefig(filename)
    
    

    




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
    
    


    
    