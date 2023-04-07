# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json
import torch
import hydra
import omegaconf
import wandb
import random
import os

from src.models.model import GFlownet
from src.utils.mols import BlockDictionary, BlockMolecule, MoleculeMDP
from src.utils.proxy import Proxy

from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt

@hydra.main(config_path='conf/', config_name="default_config.yaml", version_base="1.1")
def main(cfg):
    """
        Train model
    """
    logger = logging.getLogger(__name__)
    logger.info('train model')

    # retrieve the values from the config files
    cfg_wandb = cfg.wandb.params
    cfg_exp = cfg.experiment

    # load hyperparameters from configuration file
    epochs = cfg_exp.hp.epochs
    mbsize = cfg_exp.hp.mbsize

    device = torch.device(cfg_exp.hp.device)

    min_blocks = cfg_exp.hp.min_blocks
    max_blocks = cfg_exp.hp.max_blocks

    lr = cfg_exp.hp.lr
    beta1_adam = cfg_exp.hp.beta1_adam
    beta2_adam = cfg_exp.hp.beta2_adam
    epsilon_adam = cfg_exp.hp.epsilon_adam

    nemb = cfg_exp.hp.nemb
    num_conv_steps = cfg_exp.hp.num_conv_steps

    epsilon_loss = cfg_exp.hp.epsilon_loss

    reward_T = cfg_exp.hp.reward_T
    reward_beta = cfg_exp.hp.reward_beta

    random_action_prob = cfg_exp.hp.random_action_prob

    lambda_T = cfg_exp.hp.lambda_T
    R_min = cfg_exp.hp.R_min

    # integrate the parameters from hydra into wandb 
    # only store the experiment specific parameters
    cfg_params = omegaconf.OmegaConf.to_container(
            cfg_exp, resolve=True, throw_on_missing=True
    )["hp"]

    # initialize weights and biases agent
    wandb.init(entity=cfg_wandb.entity, project=cfg_wandb.project, name=cfg_exp.hp.run_name, config=cfg_params)
    
    bdict = BlockDictionary()

    num_out_per_stem = len(bdict.block_smis)
    num_out_per_stop = 1

    # define model
    model = GFlownet(nemb=nemb,
                     out_per_stem=num_out_per_stem,
                     out_per_stop=num_out_per_stop,
                     num_conv_steps=num_conv_steps)
    
    # define reward proxy function
    proxy = Proxy(device=device)

    # mdp used to compute parent states
    mdp = MoleculeMDP()

    # define optimizer
    opt = torch.optim.Adam(model.parameters(), lr,
                           betas=(beta1_adam, beta2_adam),
                           eps=epsilon_adam)
    leaf_losses = []
    leaf_losses_min = []
    leaf_losses_max = []
    flow_losses = []
    flow_losses_min = []
    flow_losses_max = []

    losses = []

    rewards = []

    # define training loop
    for epoch in range(epochs):
        print(epoch)
        # create a minibatch of empty molecules
        mols = [BlockMolecule() for _ in range(mbsize)]
        # mols[3].add_block(0,0)
        #[mol.add_block(0,0) for mol in mols]

        batch = [mol.to_block_graph(device=device) for mol in mols]

        mols_graph_batch = Batch.from_data_list([graph for graph in batch if graph is not None])
        mols_graph_batch.to(device)

        # check which molecules are done
        done = [0 for _ in range(mbsize)]

        # compute outflow from initial states
        out_flow_stem, out_flow_stop, stem_batch = model(mols_graph_batch)

        # make stop action probability very small when the molecule is empty
        out_flow_stop = out_flow_stop * 0 - 1000
        
        # combine flows for each molecule
        out_flow_prediction = [torch.concatenate((out_flow_stop[i],out_flow_stem[stem_batch == i].reshape(-1))) for i in range(mbsize)]

        leaf_loss = []
        flow_loss = []

        minibatch_loss = 0

        for t in range(max_blocks):
            # get actions from out flows     
            actions = [Categorical(logits=logits).sample().item() for logits in out_flow_prediction]
            
            for i in range(mbsize):
                if done[i]:
                    continue
                # with some probability choose random action - exploration
                if random.random() < random_action_prob:
                    action = random.randint(0,out_flow_prediction[i].shape[0]-1)
                else:
                    action = actions[i]
                # stop action or if the molecule cannot be expanded any further
                if (action == 0 or len(mols[i].stems) == 0) and t >= min_blocks:
                    done[i] = t
                else:
                    # every action index is shifted with 1 due to the stop action = 0
                    # therefore in order to get the correct value we subtract 1
                    action = max(0,action-1)
                    # infer the block and stem idx
                    blockidx = action % num_out_per_stem
                    stemidx = action // num_out_per_stem

                    # add block to molecule
                    mols[i].add_block(blockidx=blockidx, stemidx=stemidx)

            # update batch
            batch = [mol.to_block_graph(device=device) for mol in mols]

            mols_graph_batch = Batch.from_data_list([graph for graph in batch if graph is not None])
            mols_graph_batch.to(device)

            # compute outflow from new state
            out_flow_stem, out_flow_stop, stem_batch = model(mols_graph_batch)

            # make stop action probability very small when the molecule is too small
            if t < min_blocks:
                out_flow_stop = out_flow_stop * 0 - 1000
            
            # combine flows for each molecule
            out_flow_prediction = [torch.concatenate((out_flow_stop[i],out_flow_stem[stem_batch == i].reshape(-1))) for i in range(mbsize)]
            
            for i in range(mbsize):
                # dont include molecules that terminated in previous steps
                if done[i] != t and done[i]:
                    continue
                # compute parent states for each molecule        
                parents = mdp.parents(mols[i])

                # compute inflow from parents
                parent_flow_prediction = 0
                for parent, (blockidx, stemidx) in parents:
                    parent_graph = parent.to_block_graph(device=device)
                    parent_graph_batch = Batch.from_data_list([parent_graph])
                    parent_graph_batch.to(device)

                    in_flow_stem, _, _ = model(parent_graph_batch)

                    parent_flow_prediction += torch.exp(in_flow_stem[stemidx, blockidx])
                
                # if a molecule was done being build in this iteration compute reward
                if (done[i] == t and done[i]) or (t == max_blocks-1 and not done[i]):
                    # get reward from proxy
                    reward_true = proxy([mols[i]]).item()
                    rewards.append(reward_true)
                    # log reward
                    wandb.log({"Reward": rewards[-1]})
                    # we transform the reward as (R(x)/T)^beta
                    # make sure that R>=R_min, i.e. clip value
                    reward = (max(R_min,reward_true) / reward_T)**reward_beta
                    out_flow_prediction[i] = torch.zeros(out_flow_prediction[i].shape[0])
                else:
                    # set reward to zero
                    reward = 0

                # use loss equation from paper that utilizes log sum exponentiations
                in_flow_prediction_mol = torch.log(epsilon_loss + parent_flow_prediction)
                #import pdb
                #pdb.set_trace()
                out_flow_prediction_mol = torch.log(epsilon_loss + reward + torch.exp(out_flow_prediction[i]).sum())

                # update minibatch loss
                loss = (in_flow_prediction_mol - out_flow_prediction_mol)**2
                # multiply losses for terminal states with factor lambda
                if reward != 0:
                    loss *= lambda_T
                    leaf_loss.append(loss.item())
                else:
                    flow_loss.append(loss.item())               
                
                minibatch_loss += loss

        leaf_losses.append(np.mean(leaf_loss))
        leaf_losses_min.append(np.min(leaf_loss))
        leaf_losses_max.append(np.max(leaf_loss))
        flow_losses.append(np.mean(flow_loss))
        flow_losses_min.append(np.min(flow_loss))
        flow_losses_max.append(np.max(flow_loss))
        losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0

        wandb.log({"Leaf loss": leaf_losses[-1]})
        wandb.log({"Flow loss": flow_losses[-1]})
        wandb.log({"Loss": losses[-1]})
        wandb.watch(model)

    steps = np.arange(len(leaf_losses))
    mols = np.arange(len(rewards))

    path = f'reports/figures/{cfg_exp.hp.run_name}'
    os.makedirs(path,exist_ok=True)

    plt.figure()
    plt.plot(steps, leaf_losses, color="blue")
    plt.plot(steps, flow_losses, color="orange")
    plt.fill_between(steps, leaf_losses_min, leaf_losses_max, color="blue", alpha=0.1)
    plt.fill_between(steps, flow_losses_min, flow_losses_max, color="orange", alpha=0.1)

    plt.legend(["leaf loss", "flow loss"])

    plt.xlabel("SGD steps")
    plt.ylabel("loss")
    plt.title("Flow and leaf losses")

    filename = f"{path}/leafflowloss.png"
    plt.savefig(filename)

        
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
