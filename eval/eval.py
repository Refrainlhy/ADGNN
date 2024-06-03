import numpy as np
from deeprobust.graph.defense_pyg import GCN, SAGE
import torch.nn.functional as F
import torch, time
from copy import deepcopy
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.loader import NeighborLoader
import torch.optim as optim


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test(args, data, model_name, device):
    if(model_name.lower() == 'GCN'):
        model = GCN(nfeat=data.x.shape[1], nhid=32, nclass=data.num_classes,
            nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            device=device).to(device)
    elif(model_name.lower()=='sage'):
        model = SAGE(nfeat=data.x.shape[1], nhid=32, nclass=data.num_classes,
            nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            device=device).to(device)       
    model = model.to(device)
    model.fit(data.to(device), verbose=False) # train with earlystopping
    acc_test = model.test()
    output = model.predict()
    print("output.shape",output.shape,data.y.shape,data.target_mask.shape)
    acc_target = accuracy(output[data.target_mask], data.y[data.target_mask])
    print("Test set results:","test_acc= {:.4f}".format(acc_test),"target_acc= {:.4f} \n".format(acc_target))
    return acc_target.item(), output
