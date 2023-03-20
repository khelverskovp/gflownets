"""Proxy class for the proxy model.
   inspired by the proxy class from https://github.com/GFNOrg/gflownet/blob/master/mols/gflownet.py"""

import gzip
import pickle
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

# import MoleculeMDP from mols.py
from mols import MoleculeMDP

class MPNNet_Atom(nn.Module):
    def __init__(self, num_feat, dim,
                 num_out_per_mol,
                 num_conv_steps, dropout_rate):
        super().__init__()
        self.lin0 = nn.Linear(num_feat, dim)
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.dropout_rate = dropout_rate

        self.act = nn.LeakyReLU()

        net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')

        self.gru = nn.GRU(dim, dim)
        

        self.set2set = Set2Set(dim, processing_steps=3)

        self.lin = nn.Linear(dim * 2, num_out_per_mol)

    def forward(self, graph, do_dropout=False):
        out = self.act(self.lin0(graph.x))
        h = out.unsqueeze(0)
        h = F.dropout(h, training=do_dropout, p=self.dropout_rate)

        for i in range(self.num_conv_steps):
            m = self.act(self.conv(out, graph.edge_index, graph.edge_attr))
            m = F.dropout(m, training=do_dropout, p=self.dropout_rate)
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
            out = out.squeeze(0)

        global_out = self.set2set(out, graph.batch)
        global_out = F.dropout(global_out, training=do_dropout, p=self.dropout_rate)
        per_mol_out = self.lin(global_out) # per mol scalar outputs

        if hasattr(graph, 'nblocks'):
            per_mol_out = per_mol_out * graph.nblocks.unsqueeze(1)

        return per_mol_out

class Proxy:
    proxy_path = 'models/pretrained_proxy'

    def __init__(self, device) -> None:
        eargs = pickle.load(gzip.open(f'{self.proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{self.proxy_path}/best_params.pkl.gz'))
        self.mdp = MoleculeMDP()
        self.proxy = self.make_model(eargs)
        for a,b in zip(self.proxy.parameters(), params):
            a.data = torch.tensor(b, dtype='float64')
        self.proxy.to(device)

    def make_model(self, eargs):
        if eargs.repr_type == 'atom_graph':
            model = MPNNet_Atom(
                num_feat = eargs.num_feat, 
                num_vec = 0, 
                dim = eargs.nemb,
                out_per_mol = 1, 
                out_per_stem = eargs.num_out_per_stem,
                num_conv_steps = eargs.num_conv_steps, 
                dropout_rate = eargs.dropout_rate)
        
        return model
            
        

    def __call__(self, mol):
        mol = mol.to_atom_graph()
        return self.proxy(mol).item()
        
    
if __name__ == "__main__":
    molecule = MoleculeMDP()
    device = torch.device("cpu")

    proxy = Proxy(device=device)

    print(proxy(molecule))