from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, degree
import torch_geometric.transforms as T
import argparse
import torch, time
import deeprobust.graph.utils as utils
from deeprobust.graph.defense import ADGNN
from utils.dynamic_data_load import Dynamic_data_load
from eval.eval import test
import sys,os,json
import numpy as np
from copy import deepcopy
from influence import influence_score_computation
from utils.coreset import Coreset
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, dest='data_name', default='cora', help='data name')
parser.add_argument('--data_file', type=str, dest='data_file', default='corax', help='data file')
parser.add_argument('--gpu', type=str, default='1', help='gpu id')
parser.add_argument('--defense_model', type=str, default='GCN',help='defence model variant')
parser.add_argument('--defense_budget', type=float, default=0.1, help='active defense budget for active defense.')
parser.add_argument('--train_node_budget', type=float, default=0.1, help='the number of selected train nodes.')
parser.add_argument('--encoder', type=str, default='SAGE')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--depochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--subgraph', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--Pdist', type=int, default=2, help='p norm')
parser.add_argument('--rls_dir', type=str, default='result/', help='rls_dir')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


def get_result_path(args):
    if not os.path.exists(args.rls_dir):
        os.makedirs(args.rls_dir)
    result_path = args.rls_dir + '_'.join([args.data_name, str(args.attack_model), str(args.active_flag), 'db' ,str(args.defense_budget), 'ab', str(args.attack_budget)]) + '.json'
    return result_path


if __name__ == '__main__':

    dydata = Dynamic_data_load(args, args.data_name, args.data_file)
    print('input data: {}'.format(dydata.data))
    result, node_emb = test(args, dydata.data, args.encoder, device)
    defense_budget = int(args.defense_budget * dydata.num_nodes)
    train_node_budget = int(args.train_node_budget * len(dydata.data.train_nodes))

    t1 = time.time()
    inf_score = influence_score_computation(dydata.data, device)

    test_nodes, target_nodes = dydata.data.test_nodes, dydata.data.target_nodes
    neigh_num = degree(dydata.data.edge_index[0], dtype=torch.long)
    test_neigh_num = neigh_num[test_nodes]
    target_neigh_num = neigh_num[target_nodes]
    
    assert len(inf_score) ==  (len(dydata.data.train_nodes))
    sample_num = min(200, len(dydata.data.train_nodes)/10)
    coreset_model = Coreset(inf_score, node_emb, train_node_budget, dydata.data.train_nodes, sample_num)
    sel_train_nodes = coreset_model.select_train_coreset()

    num_nodes = dydata.data.x.shape[0]
    dydata.update_mask(defense_budget)

    active_defense_model = ADGNN(dydata.data, defense_budget, epochs=args.depochs, inf_score=inf_score, num_nodes=num_nodes, search_space_size=10000, lambda_=0.1, test_nodes = test_nodes, target_nodes= target_nodes,sel_train_nodes = sel_train_nodes, test_neigh_num = test_neigh_num, target_neigh_num = target_neigh_num, max_final_samples=3, edge_num_per_node=15, device=device)
    edge_index, edge_weight, feat = active_defense_model.active_defense()

    t2 = time.time()

    dydata.update_graph_with_events(edge_index, feat)

    print('Time:{}, generated defense data: {}'.format(t2-t1, dydata.data))

    torch.save(dydata.data, args.rls_dir + args.data_name + '_protected')






