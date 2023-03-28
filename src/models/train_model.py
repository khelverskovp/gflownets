# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json
import torch

from src.models.model import GFlownet
from src.utils.mols import BlockMolecule, MoleculeMDP
from src.utils.proxy import Proxy

from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt

@click.command()
def main():
    """
        Train model
    """
    logger = logging.getLogger(__name__)
    logger.info('train model')
    
    min_blocks = 2
    max_blocks = 8

    nemb = 256
    num_out_per_stem = 105
    num_out_per_stop = 1
    num_conv_steps = 12

    lr = 5e-4
    weight_decay = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    epochs = 500

    mbsize = 4

    device = torch.device("cpu")

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
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay,
                           betas=(beta1, beta2),
                           eps=epsilon)
    
    losses = []

    # define training loop
    for i in range(epochs):
        print(i)
        # create a minibatch of empty molecules
        mols = [BlockMolecule() for _ in range(mbsize)]
        # mols[3].add_block(0,0)
        #[mol.add_block(0,0) for mol in mols]

        batch = [mol.to_block_graph() for mol in mols]

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

        minibatch_loss = 0

        for t in range(max_blocks):
            # get actions from out flows     
            actions = [Categorical(logits=logits).sample().item() for logits in out_flow_prediction]
            
            for i in range(mbsize):
                if done[i]:
                    continue
                action = actions[i]
                # stop action or if the molecule cannot be expanded any further
                if (action == 0 or len(mols[i].stems) == 0) and t >= min_blocks:
                    done[i] = t
                else:
                    # every action index is shifted with 1 due to the stop action = 0
                    # therefore in order to get the correct value we subtract 1
                    action -= 1
                    # infer the block and stem idx
                    blockidx = action % num_out_per_stem
                    stemidx = action // num_out_per_stem

                    # add block to molecule
                    mols[i].add_block(blockidx=blockidx, stemidx=stemidx)

            # update batch
            batch = [mol.to_block_graph() for mol in mols]

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
                # compute parent states for each molecule        
                parents = mdp.parents(mols[i])

                # compute inflow from parents
                parent_flow_prediction = 0
                for parent, (blockidx, stemidx) in parents:
                    parent_graph = parent.to_block_graph()
                    parent_graph_batch = Batch.from_data_list([parent_graph])
                    parent_graph_batch.to(device)

                    in_flow_stem, _, _ = model(parent_graph_batch)

                    parent_flow_prediction += in_flow_stem[stemidx, blockidx].item()
                
                # if a molecule was done being build in this iteration compute reward
                if (done[i] == t and done[i]) or (t == max_blocks-1 and not done[i]):
                    reward = proxy([mols[i]]).item()
                    out_flow_prediction[i] = torch.zeros(out_flow_prediction[i].shape[0])
                else:
                    # set reward to zero
                    reward = 0

                minibatch_loss += (parent_flow_prediction - out_flow_prediction[i].sum() - reward)**2
        
        losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0

                    
        #print([mols[i].blockidxs for i in range(mbsize)])
        #[mols[i].draw_mol_to_file(f"{i}_molgflow",highlightBonds=True) for i in range(mbsize)]

    plt.plot(range(epochs), losses)   
    plt.show()     

        
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
