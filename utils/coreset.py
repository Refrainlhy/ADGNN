import torch
from optparse import OptionParser
import os,time,sys
import torch.optim as optim
import torch.nn as nn
import pickle,random,math
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import scipy.sparse as sp
import json
import scipy.io as sio

from torch_geometric.utils import dropout_adj, degree, to_undirected, to_dense_adj

class Coreset(object):
    """docstring for Coreset"""
    def __init__(self, inf_score, node_emb, budget, train_nodes, samp_node_num=1000):
        super(Coreset, self).__init__()
        self.inf_score = inf_score 
        self.node_emb = node_emb
        self.budget = budget
        self.train_nodes = train_nodes
        self.samp_node_num = int(samp_node_num)
        self.compute_pair_distance()
        # self.select_train_coreset()


    def compute_pair_distance(self):

        def cos_sim(a, b):
            a_norm = a.norm(dim=-1, keepdim=True)
            b_norm = b.norm(dim=-1, keepdim=True)
            a = a / (a_norm + 1e-8)
            b = b / (b_norm + 1e-8)
            return a @ b.transpose(-2, -1)

        def euc_sim(a, b):
            dist = 2 * a @ b.transpose(-2, -1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]
            return (-dist)**(1/2)

        train_emb = self.node_emb[self.train_nodes]
        self.train_node_dis = euc_sim(train_emb, train_emb)
        self.c_max = self.train_node_dis.max()

    def select_train_coreset(self):
        left_nodes = list(range(len(self.train_nodes)))
        sel_nodes = []
        for i in range(self.budget):
            if(i%100==0):
                print("i: {}".format(i))
            if(len(left_nodes)==0):
                break
            node_samp = random.sample(left_nodes, self.samp_node_num)
            node_samp = torch.tensor(node_samp)

            node_samp_inf = self.inf_score[node_samp]
            if(i==0):
                sel_max_value, sel_max_idx = node_samp_inf.max(dim=-1)

            else:
                node_samp_div, indices = self.train_node_dis[node_samp,:][:,sel_nodes].min(dim=1)
                # node_samp_div = self.c_max - node_samp_dis
                sel_max_value, sel_max_idx = (node_samp_inf + node_samp_div).max(dim=-1)

            sel_node_idx = node_samp[sel_max_idx].item()

            sel_nodes.append(sel_node_idx)
            left_nodes.remove(sel_node_idx)

        self.sel_nodes = sel_nodes
        return self.sel_nodes


