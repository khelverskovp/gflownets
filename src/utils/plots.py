import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import os

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
                leaf_losses.extend(data["leaf_losses"])
                leaf_losses_min.extend(data["leaf_losses_min"])
                leaf_losses_max.extend(data["leaf_losses_max"])
                flow_losses.extend(data["flow_losses"])
                flow_losses_min.extend(data["flow_losses_min"])
                flow_losses_max.extend(data["flow_losses_max"])
        except EOFError:
            pass

    steps = np.arange(len(leaf_losses))

    plt.figure()
    plt.loglog(steps, leaf_losses, color="blue")
    plt.loglog(steps, flow_losses, color="orange")
    plt.fill_between(steps, leaf_losses_min, leaf_losses_max, color="blue", alpha=0.1)
    plt.fill_between(steps, flow_losses_min, flow_losses_max, color="orange", alpha=0.1)

    plt.ylim(0.00001,6000)
    plt.yticks([0.0001,0.001,0.01,0.1,1,10,100,1000])

    plt.legend(["leaf loss", "flow loss"])

    plt.xlabel("SGD steps")
    plt.ylabel("loss")
    plt.title("Flow and leaf losses")

    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)

    # save file
    file_id = len(steps)
    filename = f"{figures_path}/leafflowloss_{file_id}.png"
    plt.savefig(filename)

def make_rewards_plot(experiment_id):
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass

    # make a plot of reward on y axis vs molecules generated (length of rewards list) on x axis
    rids = np.arange(len(rewards))
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

    

    file_id = rids
    figures_path = f"reports/figures/{experiment_id}"
    os.makedirs(figures_path,exist_ok=True)


    filename = f"{figures_path}/rewards_{file_id}.png"
    plt.savefig(filename)

    
