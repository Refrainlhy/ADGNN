import pathlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import torch
import time
import torch.nn.functional as F
import torchvision
from sklearn import linear_model
from torch import nn
import torch.utils 
from tqdm import tqdm, trange
from influence import BaseObjective, CGInfluenceModule, log_regression
from copy import deepcopy
from torch_geometric.nn.conv import SimpleConv


class BinClassObjective(BaseObjective):

    def train_outputs(self, model, batch):
        # print(111)
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        # print(222)
        f = nn.LogSoftmax(dim=-1)
        nll_loss = nn.NLLLoss()
        return nll_loss(f(outputs), batch[1])

    def train_regularization(self, params):
        # print(333)
        # print(params)
        L2_WEIGHT = 1e-4
        return L2_WEIGHT * torch.square(params.norm())

    def test_loss(self, model, params, batch):
#         print(len(batch[0]),len(batch[1]))
#         print(batch[0])
        outputs = model(batch[0])
        f = nn.LogSoftmax(dim=-1)
        node_idx = list(range(len(batch[1])))
        # print(params)
        return f(outputs)[node_idx, batch[1]].sum()

def influence_score_computation(data, device):
    t0 = time.time()
    x = SimpleConv(aggr='mean')(data.x,data.edge_index)
    # x = SimpleConv(aggr='mean')(x,data.edge_index)
    t1 = time.time()
    # print("Time of SimpleConv: {}".format(t1-t0))

    classifier, _ = log_regression(x, data.y, 2000, device,
                   data.train_mask,
                   data.val_mask,
                   data.test_mask)

    X_train = x[data.train_mask]
    Y_train = data.y[data.train_mask]
    train_set = torch.utils.data.TensorDataset(X_train, Y_train)

    X_test = x[data.test_mask]
    Y_test = data.y[data.test_mask]
    # X_test = torch.randn((5,20))
    # Y_test = torch.randint(0,2,(5,)).to(torch.float)
    test_set = torch.utils.data.TensorDataset(X_test, Y_test)

    t2 = time.time()
    # print("Time of MLP training: {}".format(t2-t1))

    module = CGInfluenceModule(
        model = classifier,
        objective=BinClassObjective(),
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32),
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32),
        device = device,
        damp = 0.001,
        atol = 1e-8,
        maxiter = 500,
    )

    all_train_idxs = list(range(X_train.shape[0]))
    test_idxs = list(range(X_test.shape[0]))
    inf_score = module.influences(train_idxs=all_train_idxs, test_idxs=test_idxs)
    t3 = time.time()
    print("Time of influence computation: {}".format(t3-t2))
    inf_score = (inf_score+10e-5).abs()
    return inf_score



