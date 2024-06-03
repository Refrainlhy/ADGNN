from typing import Optional
import numpy as np
import torch,random
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GINConv, SGConv, GATConv
from torch_geometric.nn import  global_add_pool
from torch.optim import Adam


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        # self.fc = nn.Linear(ft_in, nb_classes)
        self.fc1 = nn.Linear(ft_in, 50)
        self.fc2 = nn.Linear(50, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc2(self.fc1(seq))
        return ret

def evaluation(y_true_1, y_pred_1):
    y_true = y_true_1.view(-1)
    y_pred = y_pred_1.view(-1)
    total = y_true_1.size(0)
    correct = (y_true == y_pred).to(torch.float32).sum()
    return (correct / total).item()

def log_regression(z, y, num_epochs, test_device,
                   train_mask,
                   val_mask,
                   test_mask,
                   verbose: bool = False,
                   data_name='Cora'):
    test_device = z.device if test_device is None else test_device
    z = z.detach() 
    num_hidden = z.size(1)
    # if(data_name.lower()=='products'):
    #     y = dataset[0].y.argmax(-1).view(-1) 
    # else:
    #     y = dataset[0].y.view(-1) 
    num_classes =y.max().item() + 1
    # print("num_classes",num_classes)
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

   
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        
        classifier.to(test_device)
        optimizer.zero_grad()
        # train_idx_tmp = split['train']

        output = classifier(z[train_mask].to(test_device))
        loss = nll_loss(f(output), y[train_mask].to(test_device))

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            if val_mask.sum()>0:
                test_acc = evaluation(y[test_mask].view(-1, 1),classifier.to('cpu')(z[test_mask]).argmax(-1).view(-1, 1))
                val_acc = evaluation(y[val_mask].view(-1, 1),classifier.to('cpu')(z[val_mask]).argmax(-1).view(-1, 1))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                test_acc = evaluation(y[test_mask].view(-1, 1),classifier.to('cpu')(z[test_mask]).argmax(-1).view(-1, 1))
                if best_test_acc < test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')
    return    classifier, {'acc': best_test_acc}


 