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
    rewards_beta10 = []

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

    with gzip.open(f"results/experiment_1/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards_beta10.extend(pickle.load(fr))
        except EOFError:
            pass
    

    # change to numpy array
    rewards_beta1 = np.array(rewards_beta1)
    rewards_beta4 = np.array(rewards_beta4)
    rewards_beta10 = np.array(rewards_beta10)

    # get rewards for proxy dataset data/processed/rewards_proxy_dataset.pkl.gz
    df_rewards = pd.read_pickle("data/processed/rewards_proxy_dataset.pkl.gz")
    
    # change to numpy array
    df_rewards = np.array(df_rewards)

    linestyles = ['--', '-', ':', '-']
    colors = ['blue', 'blue', 'blue', 'black']
    #legends = [r'$\hat{p}(R | \beta=1)$', r'$\hat{p}(R | \beta=4)$', r'$\hat{p}(R | \beta=10)$', r'$\hat{p}(R | \mathrm{proxy\ dataset})$']
    legends = [r'ours, $\beta=1$',r'ours, $\beta=4$',r'ours, $\beta=10$',"proxy dataset"]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    # plot empirical density for each beta
    for i, rewards in enumerate([rewards_beta1, rewards_beta4, rewards_beta10, df_rewards]):
        pdf, bins = np.histogram(rewards, density=False, bins=40)
        pdf = pdf / np.sum(pdf)
        ax.plot(bins[:-1], pdf, linestyle=linestyles[i], color=colors[i], label=legends[i])
        

    ax.grid()
    ax.set_xlim(0,9)
    ax.set_ylim(bottom=0.00)

    fontsize = 16
    ax.set_xlabel(r'$R(x)$',fontsize=fontsize)
    ax.set_ylabel(r'$\hat{p}(R)$',fontsize=fontsize)
    ax.set_xticks(np.arange(0, 9, 2))
    ax.set_yticks([0.0,0.02,0.04,0.06,0.08])
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(loc="upper right",fontsize=fontsize-2)

    [x.set_linewidth(1.5) for x in ax.spines.values()]
    ax.xaxis.set_tick_params(width=2,length=5)
    ax.yaxis.set_tick_params(width=2,length=5)

    plt.tight_layout()

    figures_path = f"reports/figures"
    os.makedirs(figures_path, exist_ok=True)
    
    # save file
    filename = f"{figures_path}/empirical_density_plot.png"
    plt.savefig(filename)


# Figure 4
def make_top_k_plot(k_values, experiment_ids):
    # only allow for 3 thresholds
    assert len(k_values) == 3, "Please give 3 thresholds for plot to work!"

    top_k_avg_runs = [[] for _ in range(len(experiment_ids))]

    for i, eid in enumerate(experiment_ids):
        # path to rewards file is experiment_id/unique_rewards.pkl.gz    
        path = f'results/{eid}'
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
        

        for j, k in enumerate(k_values):
            print(i,j)
            top_k = np.sort(rewards[:k])

            top_k_avg = [np.mean(top_k)]

            for l in range(k+1,len(rewards)):
                if rewards[l] > top_k[0] and not is_sampled[smiles[l]]:
                    top_k[0] = rewards[l]
                    is_sampled[smiles[l]] = True
                top_k_avg.append(np.mean(top_k))
                top_k = np.sort(top_k)
            
            top_k_avg_runs[j].append(np.array(top_k_avg))

    linestyles = [':','--','-']
    legends = ["top 10", "top 100", "top 1000"]

    fig, ax = plt.subplots(figsize=(7.06, 4.1))

    for i, (k, ls, lg) in enumerate(zip(k_values,linestyles, legends)):
        top_k_rids = np.arange(k,len(rewards))
        # take mean over three runs
        top_k_avg = np.zeros(len(top_k_avg_runs[i][0]))
        
        for tk in top_k_avg_runs[i]:
            top_k_avg += tk
        top_k_avg /= len(top_k_avg_runs[i])
        ax.semilogx(top_k_rids, top_k_avg, ls, color="blue",label=lg)
        
    fontsize = 16
    plt.xlim(1e1,1e6)
    plt.ylim(2,8.5)
    plt.xticks([1e1,1e2,1e3,1e4,1e5,1e6])
    plt.yticks([2,4,6,8])
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.minorticks_off()
    plt.grid()
    plt.xlabel("molecules visited", fontsize=fontsize)
    plt.ylabel("avg " + r"$R$" + " of unique top " + r"$k$", fontsize=fontsize)

    plt.legend(fontsize=fontsize-2)

    [x.set_linewidth(1.5) for x in ax.spines.values()]
    ax.xaxis.set_tick_params(width=2,length=5)
    ax.yaxis.set_tick_params(width=2,length=5)

    plt.tight_layout()

    file_id = len(rewards)
    figures_path = f"reports/figures/{experiment_ids[0]}"
    os.makedirs(figures_path,exist_ok=True)
    
    # save file
    filename = f"{figures_path}/top_k_reward_plot_{file_id}.png"
    plt.savefig(filename)



# Figure 14
def make_tanimoto_plot(experiment_id, T):
    # get tanimoto counts for experiment
    tanimoto_counts = None
    with gzip.open(f"results/{experiment_id}/tanimoto_counts_{T}.pkl.gz") as fr:
        try:
            while True:
                data = pickle.load(fr)
                tanimoto_counts = data["tanimoto"]
                T = data["T"]
                S = data["S"]
                print(f"TANIMOTO WAS RUN WITH T={T} and S={S}")
        except EOFError:
            pass


    plt.xlim(0,1e6)
    plt.plot(np.arange(len(tanimoto_counts))+1, tanimoto_counts)
    
    plt.xlabel("states visited")
    # set ylabel with the actual T value
    plt.ylabel(f"# of modes with R > {T}")
    plt.grid()
    
    plt.xticks([0, 0.2*10**6, 0.4*10**6, 0.6*10**6, 0.8*10**6, 1.0*10**6])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/diverse_tanimoto_plot_{int(len(tanimoto_counts))}_{T}.png"
    plt.savefig(filename)
    


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
    
    rewards = np.array(rewards)
    smiles = np.array(smiles)
    unique_scaffold = defaultdict(lambda: True)
    is_bemis_murcko = np.zeros(len(rewards))
    convert_indices = np.arange(len(rewards))[rewards>T]
    smiles, smiles_idx = np.unique(smiles[rewards>T],return_index=True)
    print(f"Number of unique smiles: {len(smiles)}")
    start_time = time.time() # get start time
    elapsed_time = 0
    for i, (smi, smi_idx) in enumerate(zip(smiles,smiles_idx)):
        bemis_murcko = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi)))
        if unique_scaffold[bemis_murcko]:
            unique_scaffold[bemis_murcko] = False
            is_bemis_murcko[convert_indices[smi_idx]] = 1
        if i % 100 == 0:
            end_time = time.time() # get end time
            elapsed_time += end_time - start_time # calculate elapsed time
            print("Iteration ", i, " elapsed time: ", elapsed_time, " seconds")
            start_time = end_time

    # make plot with # of modes with R > T on the ylabel and states visisted on the xlabel
    plt.plot(np.arange(len(is_bemis_murcko))+1,np.cumsum(is_bemis_murcko))
    plt.xlabel("states visited")
    # set ylabel with the actual T value
    plt.ylabel(f"# of modes with R > {T}")
    plt.grid()

    plt.xticks([0, 0.2*10**6, 0.4*10**6, 0.6*10**6, 0.8*10**6, 1.0*10**6])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/diverse_bemis_murcko_plot_{int(len(rewards) / 4)}_{T}.png"
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
    with gzip.open(f"results/{experiment_id}/inflow_leaves.pkl.gz") as fr:
        try:
            while True:
                inflows.extend(pickle.load(fr))
        except EOFError:
            pass

    hp = None
    with gzip.open(f"results/{experiment_id}/losses.pkl.gz") as fr:
        try:
            while True:
                hp = pickle.load(fr)["hp"]
                break
        except EOFError:
            pass

    reward_T = hp["reward_T"]
    reward_beta = hp["reward_beta"]
    R_min = hp["R_min"]
    
    rewards = np.array(rewards)[-5000:]
    rewards = (np.clip(rewards, R_min, np.max(rewards)) / reward_T)**reward_beta
    inflows = np.array(inflows)[-5000:]

    fig, ax = plt.subplots(figsize=(6.97, 4.1))

    # make scatter plot
    ax.scatter(rewards,inflows, s=7, alpha=0.3)

    # make bin average
    steps = []
    bin_avg = []
    nbins = 4
    
    for i in range(-5,0):
        # create bin
        left = 10**i
        right = 10**(i+1)
        bins = np.linspace(left, right, nbins+1)
        for j in range(nbins):
            steps.append((bins[j]+bins[j+1]) / 2)
            # inflows for rewards with R > left and R < right
            ifl = inflows[(rewards > bins[j]) & (rewards < bins[j+1])]
            if len(ifl) == 0:
                if len(bin_avg):
                    bin_avg.append(bin_avg[-1])
                else:
                    steps.pop()
                continue
            bin_avg.append(np.mean(ifl))

    ax.plot(steps,bin_avg, "purple",label="bin average", linewidth=2)

    # plot x = y
    x = np.linspace(1e-4,1,10000)
    y = x
    ax.plot(x,y,"k", label = r"$x=y$", linewidth=2)

    # make logarithmic fit
    a,r = np.polyfit(np.log(rewards), np.log(inflows), 1)
    x = np.linspace(1e-4,1,10000)
    y = np.exp(r + a*np.log(x))
    l = "log-log linear regression\n"+r"$a=$"+f"{round(a,2)}" + " " + r"$r=$"+f"{round(r,2)}"
    ax.plot(x,y,"orange",label=l, linewidth=2)

    fontsize = 12

    ax.legend(loc="lower right",fontsize=fontsize)
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    
    plt.xlabel("score",fontsize=fontsize)
    plt.ylabel("predicted unnormalized probability",fontsize=fontsize)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1e-4,2)
    plt.ylim(1e-4,2)

    plt.xticks([1e-4,1e-3,1e-2,1e-1,1])
    plt.yticks([1e-4,1e-3,1e-2,1e-1,1])

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path, exist_ok=True)

    # save file
    filename = f"{figures_path}/scatter_inflow_reward_plot.png"
    plt.savefig(filename)

# Figure 17
def make_empirical_density_inflow_reward_plot(experiment_id):
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    inflows = []
    with gzip.open(f"results/{experiment_id}/inflow_leaves.pkl.gz") as fr:
        try:
            while True:
                inflows.extend(pickle.load(fr))
        except EOFError:
            pass

    outflows = []
    with gzip.open(f"results/{experiment_id}/outflow_source.pkl.gz") as fr:
        try:
            while True:
                outflows.extend(pickle.load(fr))
        except EOFError:
            pass

    hp = None
    with gzip.open(f"results/{experiment_id}/losses.pkl.gz") as fr:
        try:
            while True:
                hp = pickle.load(fr)["hp"]
                break
        except EOFError:
            pass

    reward_T = hp["reward_T"]
    reward_beta = hp["reward_beta"]
    R_min = hp["R_min"]
        
    rewards = np.array(rewards)[-10000:]
    rewards = (np.clip(rewards, R_min, np.max(rewards)) / reward_T)**reward_beta
    inflows = np.array(inflows)[-10000:]
    outflows = np.array(outflows)[-10000:]
    linestyles = ['--', '-']
    colors = ['blue', 'blue']
    legends = ["rewards","inflows"] 
    
    fig, ax1 = plt.subplots(figsize=(6.75,4.1))
    ax2 = ax1.twinx()
    
    nbins = [340,340]
    # plot empirical density for each beta
    for i, (vals, n) in enumerate(zip([rewards / outflows,inflows / outflows],nbins)):
        _, bins = np.histogram(vals, bins=n)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        pdf, bins = np.histogram(vals, density=False, bins=logbins)
        pdf = pdf / np.sum(pdf)
        
        ax1.plot(bins[:-1], pdf, linestyle=linestyles[i], color=colors[i], label=legends[i])
        ax2.plot(bins[:-1], pdf, linestyle=linestyles[i], color=colors[i], label=legends[i])
        

    ax1.grid()
    ax1.set_xlabel("predicted " + r'$\hat{p}(x)/Z$' + " and " + r'$\hat{R}(x)/Z$' )
    ax1.set_ylabel("empirical frequency of " + r'$\hat{p}(x)/Z$')
    ax2.set_ylabel("empirical frequency of " + r'$\hat{R}(x)/Z$')
    plt.xscale("log")
    
    plt.xticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
    plt.xlim(1e-8,3e-1)
    ax1.set_ylim(0,0.045)
    ax2.set_ylim(0,0.045)
    ax1.set_yticks([0,0.01,0.02,0.03,0.04])
    ax2.set_yticks([0,0.01,0.02,0.03,0.04])
    ax1.legend()
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_linestyle("--")
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)

    plt.tight_layout()

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

    
    
    fig, ax = plt.subplots(figsize=(6.64,4.1))
    ax.loglog(steps, lls, color="blue")
    ax.loglog(steps, fls, color="orange")
    ax.fill_between(steps, llsmin, llsmax, color="blue", alpha=0.2)
    ax.fill_between(steps, flsmin, flsmax, color="orange", alpha=0.2)

    [x.set_linewidth(1.5) for x in ax.spines.values()]

    plt.ylim(0.0001,2100)
    plt.yticks([0.001,0.01,0.1,1,10,100,1000])

    plt.legend(["leaf loss", "flow loss"])

    fontsize = 12
    ax.set_xlabel("SGD steps",fontsize=fontsize)
    ax.set_ylabel("loss",fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()

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
    
    


    
    