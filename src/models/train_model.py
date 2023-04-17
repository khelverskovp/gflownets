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
import pickle
import gzip
import time
import pdb

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
    total_epochs = cfg_exp.hp.total_epochs
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

    run_name = cfg_exp.hp.run_name

    # how often should the model be saved
    save_freq = cfg_exp.hp.save_freq

    # integrate the parameters from hydra into wandb 
    # only store the experiment specific parameters
    cfg_params = omegaconf.OmegaConf.to_container(
            cfg_exp, resolve=True, throw_on_missing=True
    )["hp"]

    # initialize weights and biases agent
    # wandb.init(entity=cfg_wandb.entity, project=cfg_wandb.project, name=run_name, config=cfg_params)
    
    # make sure folders are created for specific experiment
    models_path = f"models/runs/{run_name}"
    os.makedirs(models_path,exist_ok=True)
    results_path = f"results/{run_name}"
    os.makedirs(results_path,exist_ok=True)

    # instantiate model and proxy
    bdict = BlockDictionary()

    num_out_per_stem = len(bdict.block_smis)
    num_out_per_stop = 1

    # define model
    model = GFlownet(nemb=nemb,
                     out_per_stem=num_out_per_stem,
                     out_per_stop=num_out_per_stop,
                     num_conv_steps=num_conv_steps)
    
    # get the latest model checkpoint - if none simply start from scratch
    param_id = 0
    for i in range(save_freq,total_epochs+1,save_freq):
        if os.path.isfile(f"{models_path}/params_{i}.pkl.gz"):
            param_id = i
    
    # if we have a checkpoint load parameters
    if param_id:
        # if we already have a model for the specific experiment that has been fully trained
        # dont train further
        if param_id == total_epochs:
            logger.info("TRAINING IS COMPLETE!")
            time.sleep(10000)
            raise Exception("TRAINING IS COMPLETE!")
        
        params = pickle.load(gzip.open(f"{models_path}/params_{param_id}.pkl.gz"))
        for a,b in zip(model.parameters(), params):
            a.data = torch.tensor(b, dtype=torch.double)

    model.to(torch.double)
    model.to(device)
    
    # define reward proxy function
    proxy = Proxy(device=device)

    # mdp used to compute parent states
    mdp = MoleculeMDP()

    # define optimizer
    opt = torch.optim.Adam(model.parameters(), lr,
                           betas=(beta1_adam, beta2_adam),
                           eps=epsilon_adam)
    
    # make lists to store losses
    losses = []

    term_losses = []
    term_losses_min = []
    term_losses_max = []
    flow_losses = []
    flow_losses_min = []
    flow_losses_max = []

    # list to store rewards, trajectories and smiles strings
    rewards = []
    trajectories = []
    smiles = []
    
    

    # define training loop
    for epoch in range(save_freq):
        if (epoch+param_id) % 100 == 0:
            logger.info(epoch+param_id)

        # keep track of terminal and internal transition losses
        term_loss = 0
        flow_loss = 0

        # keep track of max and min term and flow losses
        min_term_loss = np.inf
        max_term_loss = -np.inf
        min_flow_loss = np.inf
        max_flow_loss = -np.inf
        
        # number of terminal transitions will be mbsize
        n_term_transitions = mbsize

        # number of internal transitions will be variable - keep counter
        n_flow_transitions = 0

        for i in range(mbsize):
            # create empty molecule
            mol = BlockMolecule()

            # keep track of trajectory
            traj = []

            # boolean to check if we are done
            terminal_state = False
            
            # get initial output from model
            # turn into block graph batch format
            graph = mol.to_block_graph(device=device)

            mols_graph_batch = Batch.from_data_list([graph])
            mols_graph_batch.to(device)

            # get output from model
            out_flow_stem, out_flow_stop, _ = model(mols_graph_batch)
            
            # loop over trajectory
            for t in range(max_blocks):
                # make probability of taking stop action very small for t < min_blocks
                if t < min_blocks:
                    out_flow_stop = out_flow_stop * 0 - 1000

                # put unnormalized log probabilities into list
                logits = torch.concatenate([out_flow_stop.reshape(-1), out_flow_stem.reshape(-1)])
                
                # choose action based on logits
                action = Categorical(logits=logits).sample().item()
                    
                # take random action with probability random_action_prob - exploration
                if random.random() < random_action_prob:
                    # only include stop action 0 if t >= min_blocks
                    action = random.randint(int(t < min_blocks),logits.shape[0]-1)
                
                # check if we choose to stop
                if action == 0 and t >= min_blocks:
                    # compute reward from proxy
                    reward_true = proxy([mol])
                    # match reward to specific molecule
                    rewards.append(reward_true.cpu().item())
                    # we transform the reward as (R(x)/T)^beta
                    # make sure that R>=R_min, i.e. clip value
                    reward = (max(R_min,reward_true.item()) / reward_T)**reward_beta

                    # if we chose to stop the only incoming flow will be out_flow_stop
                    in_flow = torch.exp(out_flow_stop[0])

                    # outflow is zero for terminal states
                    out_flow = torch.tensor([0], device=device)

                    # add stop action to trajectory
                    traj.append((-1,0))

                    # add smiles string
                    smiles.append(mol.get_smiles())

                    # state that we are done
                    terminal_state = True
                else:
                    # execute action
                    # every action index is shifted with 1 due to the stop action = 0
                    # therefore in order to get the correct value we subtract 1
                    action = max(0,action-1)
                    # infer the block and stem idx
                    blockidx = action % num_out_per_stem
                    stemidx = action // num_out_per_stem

                    # add block to molecule
                    mol.add_block(blockidx=blockidx, stemidx=stemidx)

                    # add action to trajectories
                    traj.append((blockidx,stemidx))

                    # check if we are forced to stop
                    if len(mol.blockidxs) > 0 and (len(mol.stems) == 0 or t == max_blocks-1):
                        # compute reward from proxy
                        reward_true = proxy([mol])
                        # match reward to specific molecule
                        rewards.append(reward_true.cpu().item())
                        # we transform the reward as (R(x)/T)^beta
                        # make sure that R>=R_min, i.e. clip value
                        reward = (max(R_min,reward_true.item()) / reward_T)**reward_beta

                        # compute parent states for each molecule        
                        parents = mdp.parents(mol)

                        # if we are forced to stop the inflow will be sum of flows from all the parents
                        # compute inflow from parents
                        in_flow = 0
                        for parent, (blockidx, stemidx) in parents:
                            # turn into block graph
                            parent_graph = parent.to_block_graph(device=device)
                            parent_graph_batch = Batch.from_data_list([parent_graph])
                            parent_graph_batch.to(device)

                            # compute out flow from parent
                            out_flow_parent, _, _ = model(parent_graph_batch)

                            # add the flow for the specific action and do exp(result) because the model outputs logits
                            in_flow += torch.exp(out_flow_parent[stemidx, blockidx])

                        # outflow is zero for terminal states
                        out_flow = torch.tensor([0], device=device)
                        
                        # add smiles string
                        smiles.append(mol.get_smiles())

                        # state that we are done
                        terminal_state = True
                    else:
                        # reward is 0 for internal states
                        reward = 0

                        # compute parent states for each molecule        
                        parents = mdp.parents(mol)

                        # compute inflow from parents
                        in_flow = 0
                        for parent, (blockidx, stemidx) in parents:
                            # turn into block graph
                            parent_graph = parent.to_block_graph(device=device)
                            parent_graph_batch = Batch.from_data_list([parent_graph])
                            parent_graph_batch.to(device)

                            # compute out flow from parent
                            out_flow_parent, _, _ = model(parent_graph_batch)

                            # add the flow for the specific action and do exp(result) because the model outputs logits
                            in_flow += torch.exp(out_flow_parent[stemidx, blockidx])

                        # compute out flow
                        # turn into block graph batch format
                        graph = mol.to_block_graph(device=device)

                        mols_graph_batch = Batch.from_data_list([graph])
                        mols_graph_batch.to(device)

                        # get output from model
                        out_flow_stem, out_flow_stop, _ = model(mols_graph_batch)

                        # out_flow is simply the sum of the model outputs
                        out_flow = torch.exp(out_flow_stem).sum() + torch.exp(out_flow_stop).sum()

                # compute log of inflows and outflows with log
                in_flow = torch.log(epsilon_loss + in_flow)
            
                out_flow = torch.log(epsilon_loss + reward + out_flow)

                # compute squared difference
                loss = (in_flow - out_flow).pow(2)

                # multiply loss with lambda if terminal state
                if terminal_state:
                    tl = loss * lambda_T
                    term_loss += tl
                    
                    # update min and max values
                    min_term_loss = min(min_term_loss, tl.cpu().item())
                    max_term_loss = max(max_term_loss, tl.cpu().item())
                    break
                else:
                    fl = loss
                    flow_loss += fl

                    # update min and max values
                    min_flow_loss = min(min_flow_loss, fl.cpu().item())
                    max_flow_loss = max(max_flow_loss, fl.cpu().item())

                    # add to internal transition counter
                    n_flow_transitions += t
                
            # add trajectory to list
            trajectories.append(traj)
                
        # take average of terminal and internal flow loss
        term_loss /= n_term_transitions
        flow_loss /= n_flow_transitions

        # compute total minibatch loss
        minibatch_loss = term_loss + flow_loss

        opt.zero_grad()
        minibatch_loss.backward()
        opt.step()

        # log losses
        losses.append(minibatch_loss.cpu().item())

        # log terminal and flow losses
        term_losses.append(term_loss.cpu().item())
        term_losses_min.append(min_term_loss)
        term_losses_max.append(max_term_loss)
        flow_losses.append(flow_loss.cpu().item())
        flow_losses_min.append(min_flow_loss)
        flow_losses_max.append(max_flow_loss)


    logger.info("Beginning saving process!")

    # save model
    new_param_id = param_id + save_freq
    pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f"{models_path}/params_{new_param_id}.pkl.gz", 'wb'))
    
    # log rewards and trajectories
    pickle.dump(rewards,
                gzip.open(f"{results_path}/rewards.pkl.gz", 'ab'))

    pickle.dump(trajectories,
                gzip.open(f"{results_path}/trajectories.pkl.gz", 'ab'))
    
    pickle.dump(smiles,
                gzip.open(f"{results_path}/smiles.pkl.gz", 'ab'))
    
    # log losses
    pickle.dump({'losses': losses,
                 'term_losses': term_losses,
                 'term_losses_min': term_losses_min,
                 'term_losses_max': term_losses_max,
                 'flow_losses': flow_losses,
                 'flow_losses_min': flow_losses_min,
                 'flow_losses_max': flow_losses_max,
                 'param_id:': param_id,
                 'hp': cfg_params},
                    gzip.open(f'{results_path}/losses.pkl.gz', 'ab'))
    logger.info("Done saving!")

    time.sleep(10)

        
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
