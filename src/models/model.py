import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

from src.utils.mols import BlockDictionary, BlockMolecule

# heavily inspired by code from https://github.com/GFNOrg/gflownet/blob/master/mols/model_block.py

class GFlownet(nn.Module):
    
    def __init__(self, nemb, out_per_stem, out_per_stop, num_conv_steps):
        super().__init__()

        # block dictionary
        self.bdict = BlockDictionary()
        
        # define embeddings for each block, stem and bond
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.bdict.n_unique_blocks + 1, nemb),
            nn.Embedding(self.bdict.n_stem_types + 1, nemb),
            nn.Embedding(self.bdict.n_stem_types, nemb)
        ])

        # define activation function
        self.act = nn.LeakyReLU()

        # output values for each block in the first layer
        self.block2emb = nn.Sequential(nn.Linear(nemb, nemb), self.act,
                                        nn.Linear(nemb, nemb))

        # graph convolutional layer
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr="mean")
        self.gru = nn.GRU(nemb, nemb)

        # output values for each stem
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2, nemb), self.act,
                                       nn.Linear(nemb, nemb), self.act,
                                       nn.Linear(nemb, out_per_stem))
        
        # output value for stop action
        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), self.act,
                                         nn.Linear(nemb, out_per_stop))
        
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb

    def forward(self, graph):
        # retrieve embeddings for blocks, stems and bonds
        # blockemb has shape (73, nemb), i.e. an embedding for each block (73 -> empty block)
        # stememb has shape (215, nemb), i.e. an embedding for each stem (214 -> empty stem)
        # bondemb has shape (214, nemb), i.e. an embedding for each stem in a junction bond
        blockemb, stememb, bondemb = self.embeddings
        
        # get embeddings for the blocks, stems and bonds in the block graph
        block_emb = blockemb(graph.x) # has shape (numblocks, nemb)
        stem_emb = stememb(graph.stemtypes) # has shape (num_jbonds, nemb)
        bond_emb = bondemb(graph.edge_attr) # has shape (num_jbonds, 2, nemb)
        
        # multiply embeddings for each stem embedding in the junction bonds
        # bond_emb[:, 0][:,:,None] has shape (num_jbonds, nemb, 1), i.e. an embedding the first block stem in each bond
        # bond_emb[:, 1][:,None,:] has shape (num_jbonds, 1, nemb), i.e. an embedding the second block stem in each bond
        # bond_emb[:, 0][:,:,None] * bond_emb[:, 1][:,None,:] multiplies embeddings together between
        # corresponding stem embeddings for each jbond
        bond_emb = bond_emb[:, 0][:,:,None] * bond_emb[:, 1][:,None,:] # has shape (num_jbonds, nemb, nemb)

        # reshape to make an embedding for each junction bond
        bond_emb = bond_emb.reshape((bond_emb.shape[0],self.nemb**2)) # has shape (num_jbonds, nemb^2)

        # get output from first layer 
        out = self.block2emb(block_emb) # has shape (numblocks, nemb)

        # get a copy with extra dimension for gnn
        h = out.unsqueeze(0) # has shape (1, numblocks, nemb)

        # graph convolutional logic layer
        for _ in range(self.num_conv_steps):
            # retrieve message
            # graph.edge_index has shape (2, num_jbonds)
            m = self.act(self.conv(out, graph.edge_index, bond_emb)) # has shape (numblocks, nemb)
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            # out has shape (1, numblocks, nemb) and h has shape (1, numblocks, nemb)
            out = out.squeeze(0) # has shape (numblocks, nemb)
        
        # get batch start indices for each block and stem
        block_batch_startidx = graph._slice_dict["x"][:-1]
        # gives the starting index for each batch
        # e.g. lets assume stems looks like the following
        # stems = [[2,1], <- batch idx 0
        #          [2,3], <- batch idx 0
        #          [3,0], <- batch idx 0
        #          [2,1], <- batch idx 1
        #          [2,3], <- batch idx 1
        #          [3,0]] <- batch idx 1
        # then stem_batch_startidx = [0,3,6]
        stem_batch_startidx = graph._slice_dict["stems"]
        # this maps from [0,3,6] to [0,0,0,1,1,1]
        stem_batch = np.concatenate([i * np.ones(stem_batch_startidx[i+1] - stem_batch_startidx[i]) for i in range(len(stem_batch_startidx)-1)])
        stem_batch = np.asarray(stem_batch, dtype=np.int32)

        # get corresponding block idx for each stem
        stem_block_batch_idx = [(block_batch_startidx[stem_batch[i]] + block_idx).item() for i, block_idx in enumerate(graph.stems[:,0])]

        # convert to tensort
        stem_block_batch_idx = torch.tensor(stem_block_batch_idx)
        
        # concatenate embedding for each open stem with the corresponding output from the convolutional layer
        stem_out_cat = torch.cat([out[stem_block_batch_idx], stem_emb], 1) # has shape (num_stems, 2*nemb)

        # get output for each stem (out_per_mol long vector)
        # simple neural network with some hidden layers
        stem_pred = self.stem2pred(stem_out_cat) # has shape (num_stems, out_per_stem) (105 in our case)

        # compute value for stop action
        stop_pred = self.global2pred(gnn.global_mean_pool(out, graph.batch)) # has shape (batchsize, 1)
        
        return stem_pred, stop_pred, stem_batch
    
if __name__ == "__main__":
    molecule = BlockMolecule()

    # build molecule
    # choose molecules to add
    block_list = [91, 29, 48, 95]

    for block in block_list:
        try:
            molecule.add_block(block)
        except:
            break

    molecule.jbonds = [[0, 1, 3, 0], [0, 2, 6, 4], [2, 3, 0, 3]]

    mol_graph = molecule.to_block_graph()

    """ mol_new = BlockMolecule()
    mol_new.add_block(0)
    mol_new.add_block(0)

    print(mol_new.stems)

    mol_new_graph = mol_new.to_block_graph() """

    mol_graph_batch = Batch.from_data_list([graph for graph in [mol_graph] if graph is not None])
    mol_graph_batch.to(torch.device("cpu"))

    net = GFlownet(256, 105, 1, 12)
    
    stem_out, stop_out = net(mol_graph_batch)

    logits = torch.cat([stop_out.reshape(-1), stem_out.reshape(-1)])
    
    cat = torch.distributions.Categorical(
            logits=logits)
    action = cat.sample().item()
    
    if action == 0:
        print("STOPPPPPPPPPPPPPPPPPPPPPPPPPPP!")
    else:
        action -= 1
        blockidx = action % 105
        stemidx = action // 105
        print(f"Choose block: {blockidx}\nChoose stem: {stemidx}")
        molecule.draw_mol_to_file("before",highlightBonds=True)
        molecule.add_block(blockidx, stemidx)
        molecule.draw_mol_to_file("after",highlightBonds=True)



    
