"""Proxy class for the proxy model.
   inspired by the proxy class from https://github.com/GFNOrg/gflownet/blob/master/mols/gflownet.py"""

import gzip
import pickle
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
import requests

# import MoleculeMDP from mols.py
from mols import BlockMolecule, MoleculeMDP

from chem import atomic_numbers

from torch_sparse import coalesce

from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

import os

import pandas as pd

import json

class MPNNet_Atom(nn.Module):
    def __init__(self, num_feat, dim, num_conv_steps, num_out_per_mol=1, dropout_rate=0):
        super().__init__()

        """
        params:
        num_feat: number of input features
        dim: number of dimension in each hidden layers
        num_conv_steps: number of convolutional steps
        num_out_per_mol: number of output features
        dropout_rate: dropout rate
        """
        # 14 + use_stem_mask (always true) + 56
        # REMEMBER TO COMMENT THIS BETTER!!
        # num_feat = 71

        # since we are only interested in a single scalar reward
        # num_out_per_mol = 1, when used to evaluate the proxy
        self.num_out_per_mol = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.dropout_rate = dropout_rate

        # activation function
        self.act = nn.LeakyReLU()

        # first layer - simple layer from num_feat to dim dimensions
        self.lin0 = nn.Linear(num_feat, dim)

        # defines graph convolutional layer
        net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')

        # defines structure for GRU layer
        self.gru = nn.GRU(dim, dim)
        
        # set2set aggregation
        self.set2set = Set2Set(dim, processing_steps=3)

        # last linear linear - ensures that one output is given for each molecule in the batch
        self.lin1 = nn.Linear(dim * 2, self.num_out_per_mol)

        

    def forward(self, data, do_dropout=False):
        # natomsbatch = number of atoms in total for all molecules in batch
        # data.x has shape (natomsbatch, num_feat)
        out = self.act(self.lin0(data.x)) # out has shape (natomsbatch, dim)
        h = out.unsqueeze(0) # h has shape (1, natomsbatch, dim)
        h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
        
        for i in range(self.num_conv_steps):
            # num_edges: number of edges/bonds in batch
            # num_bond_features: number of bond types (4, because single, double, triple and aromatic)
            # data.edge_index has shape (2, num_edges)
            # data.edge_attr has shape (num_edges, num_bond_features)
            m = self.act(self.conv(out, data.edge_index, data.edge_attr)) # m has shape (natomsbatch, dim)
            m = F.dropout(m, training=do_dropout, p=self.dropout_rate)
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            # after the gru layer:
            # out has shape (1, natomsbatch, dim)
            # h has shape (1, natomsbatch, dim)
            h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
            out = out.squeeze(0) # out has shape (natomsbatch, dim)

        # data.batch contains batchidxs for each atom in the graph
        # e.g. batch consists of two molecules of natomsize 8 and 6 respectively
        # then natomsbatch=8+6=14
        # and data.batch = [0,0,0,0,0,0,0,0,1,1,1,1,1,1]
        # data.batch.shape = (natomsbatch)
        global_out = self.set2set(out, data.batch) # global_out has dim (num_mols_in_batch, 2*dim)
        global_out = F.dropout(global_out, training=do_dropout, p=self.dropout_rate)

        per_mol_out = self.lin1(global_out) # per_mol_out has shape (num_mols_in_batch, 1)
        # i.e. a reward is computed for each molecule in the batch

        return per_mol_out

# heavily inspired by code from https://github.com/recursionpharma/gflownet
class Proxy:
    proxy_path = 'models/pretrained_proxy'

    def __init__(self, device, dim=64, num_conv_steps=12):
        # load model parameters from pretrained proxy
        params = pickle.load(gzip.open(f'{self.proxy_path}/best_params.pkl.gz'))

        self.device = device

        self.nhid = 64
        self.num_conv_steps = num_conv_steps

        # make model
        self.mpnn = self.make_model(params)

    def make_model(self, params):
        # number of features for each atom in the molecule
        # one hot encoded
        # the first 14 values will contain information about hybridization type, donor/acceptor type, etc.
        # the extra 1 is used to create a stem mask
        # the remaining 56 entries will denote which atom it is
        num_feat = (14 + 1 + len(atomic_numbers))
        mpnn = MPNNet_Atom(num_feat=num_feat, dim=self.nhid, num_conv_steps=self.num_conv_steps)
        
        # load only parameters that appears in the pretrained proxy
        param_map = {
            'lin0.weight': params[0],
            'lin0.bias': params[1],
            'conv.bias': params[3],
            'conv.nn.0.weight': params[4],
            'conv.nn.0.bias': params[5],
            'conv.nn.2.weight': params[6],
            'conv.nn.2.bias': params[7],
            'conv.lin.weight': params[2],
            'gru.weight_ih_l0': params[8],
            'gru.weight_hh_l0': params[9],
            'gru.bias_ih_l0': params[10],
            'gru.bias_hh_l0': params[11],
            'set2set.lstm.weight_ih_l0': params[16],
            'set2set.lstm.weight_hh_l0': params[17],
            'set2set.lstm.bias_ih_l0': params[18],
            'set2set.lstm.bias_hh_l0': params[19],
            'lin1.weight': params[20],
            'lin1.bias': params[21],
        }
        # set values appropriately
        for k, v in param_map.items():
            mpnn.get_parameter(k).data = torch.tensor(v)
        return mpnn
            
    def compute_reward(self, mols):
        # convert each molecule to an atomic graph
        graphs = [mol.to_atom_graph() for mol in mols]
        # check validity for each graph
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        # if every graph is None say the input batch is invalid
        if not is_valid.any():
            return torch.zeros((0, 1)), is_valid
        
        # make into batch for all valid graphs
        batch = Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)

        # compute model predictions
        preds = self.mpnn(batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0

        # limit to values between 1e-4 and 100 (ensures positive rewards only)
        preds = preds.clip(1e-4, 100).reshape((-1, 1))
        return preds, is_valid
    
    def __call__(self, mols):
        preds, is_valid = self.compute_reward(mols)
        assert is_valid.any(), "Invalid graphs"
        return preds


    
    
        
if __name__ == "__main__":
    molecule = BlockMolecule()
    device = torch.device("cpu")

    block_list = [96, 36, 93]

    for block in block_list:
        try:
            molecule.add_block(block)
        except:
            break

    molecule.jbonds = [[0, 1, 3, 6], [0, 2, 8, 0]]
    molecule.stems = []

    #proxy = Proxy(device=device)

    """ for a in proxy.proxy.parameters():
        if len(a.data.shape) == 1:
            a.data = a.data.unsqueeze(0) """

    proxy = Proxy(device)

    # load data points
    filename = "docked_mols.csv"
    path = f"data/processed/{filename}"

    df = pd.read_csv(path)

    columns = df.columns
    
    # the remaining columns contains lists of numbers
    # they are in string form however in the native dataframe
    # should be converted to list type 
    for name in columns[2:]:
        df.loc[:,name] = df[name].apply(json.loads)

    # lambda function to transform dockscore into same values as proxy rewards
    mu, std = [-8.6, 1.1]
    transform = lambda v: 4 - (v-mu) / std

    avg_err = 0

    for i in range(1000):
        if i % 100 == 0:
            print(i)
        # build molecule
        mol = BlockMolecule()

        for block in df.blockidxs[i]:
            mol.add_block(block)
            
        mol.jbonds = df.jbonds[i]
        mol.stems = df.stems[i]

        pred = proxy([mol])[0].item()
        target = transform(df.dockscore[i])

        avg_err += abs(pred - target)
    
    print(avg_err / 1000)
