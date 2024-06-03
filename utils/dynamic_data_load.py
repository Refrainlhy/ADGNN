import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Yelp, AmazonProducts
from torch_geometric.utils import degree, to_undirected, add_self_loops
import torch_geometric.transforms as T
import argparse
from torch_geometric.data import Data
import torch, time
import numpy as np
from copy import deepcopy
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader


class Dynamic_data_load(object):
    """docstring for Dynamic_data_load"""
    def __init__(self, args, data_name, data_file):
        super(Dynamic_data_load, self).__init__()
        self.args = args
        self.data_name = data_name
        self.data = torch.load(data_file)
        self.data.num_classes = int(self.data.y.max().item() + 1)

        self.num_nodes = self.data.x.shape[0]
        self.num_edges = self.data.edge_index.shape[-1]

        if("train_nodes" not in self.data):
            self.data.train_nodes = self.mask_to_index(self.data.train_mask, self.num_nodes)
            self.data.val_nodes = self.mask_to_index(self.data.val_mask, self.num_nodes)
            self.data.test_nodes = self.mask_to_index(self.data.test_mask, self.num_nodes)
            self.data.target_nodes = self.mask_to_index(self.data.target_mask, self.num_nodes)

        self.num_target_nodes = len(self.data.target_nodes)
        
    def update_graph_injective(self, edge_index, x):
        self.data.edge_index = edge_index
        self.data.x = x
        self.cur_num_nodes = self.data.x.shape[0]

   
    def mask_to_index(self, index, size):
        all_idx = np.arange(size)
        return all_idx[index]

    def update_mask(self, defense_budget):
        self.cur_num_nodes = self.data.x.shape[0] + defense_budget
        self.data.train_mask = self.index_to_mask(self.data.train_nodes, self.cur_num_nodes)
        self.data.val_mask = self.index_to_mask(self.data.val_nodes, self.cur_num_nodes)
        self.data.test_mask = self.index_to_mask(self.data.test_nodes, self.cur_num_nodes)
        self.data.target_mask = self.index_to_mask(self.data.target_nodes, self.cur_num_nodes)
        self.data.node_raw_id = torch.tensor(range(self.cur_num_nodes))
        self.data.y = torch.cat([self.data.y, torch.zeros(defense_budget)]).to(torch.long)

    def index_to_mask(self, index, size):
        mask = torch.zeros((size, ), dtype=torch.bool)
        mask[index] = 1
        return mask

    def update_graph_with_events(self, edge_index, feat):
        self.data.x = feat
        self.data.edge_index = edge_index
        self.data.edge_index, _  = add_self_loops(self.data.edge_index)
        self.data.edge_index =  to_undirected(self.data.edge_index)
        self.data.num_nodes = self.data.x.shape[0]






 