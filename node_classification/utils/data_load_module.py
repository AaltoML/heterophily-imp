import os, json, pickle
from torch_geometric import datasets
from torch_geometric.utils import homophily, to_edge_index
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
# from sklearn.model_selection import StratifiedShuffleSplit 
from torch.utils.data import random_split
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = f'{ROOT_DIR}/data'

Datasets = {'CiteNet': ["Cora", "CiteSeer", "PubMed"],
            'WebKB': ["Cornell", "Texas", "Wisconsin"],
            'Amazon': ['Computers', 'Photo'],
            'WikipediaNet':  ['Chameleon', 'Squirrel' ],
            'HetGraph' : ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions'],
            'CoauthorCS': ['CoauthorCS']
            }


_dataset_list = [ ds  for dss in Datasets.values() for ds in dss]

def get_subsets(ds):
    return Datasets.get(ds, None)

def search_dataset_name(name):
    for ds, names in Datasets.items():
        if name in names:
            return ds
    return None

def data_load(name, **kwargs):
    root = DATA_PATH
    if not os.path.exists(root):
        print(f'dataset path [{root}] not exist, build a new one!')
        os.makedirs(root)
    ds = search_dataset_name(name)
    assert ds is not None, f"The data name [{name}] is not included!"
    if ds == 'CiteNet':
        data = datasets.Planetoid(root = root, name = name, split = 'geom-gcn', transform = T.ToSparseTensor())
    elif ds == 'WebKB':
        data = datasets.WebKB(root = root, name = name, transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()] ) )
    elif ds == 'Amazon':
        data =  datasets.Amazon(root = root, name= name, transform = T.ToSparseTensor() )
    elif ds == 'WikipediaNet':
        data = datasets.WikipediaNetwork(root, name, transform = T.ToSparseTensor())
    elif ds == 'HetGraph':
        data = datasets.HeterophilousGraphDataset(root, name, transform = T.ToSparseTensor() )
    elif ds == 'CoauthorCS':
        data = datasets.Coauthor(root, 'CS', transform = T.ToSparseTensor())
    # elif name == 'Actor':
    #     ds = datasets.Actor(root, transform = T.ToSparseTensor())

    return data

def data_split(
        seed: int,
        graphs,
        data_name: str = None
        ):
    split_seed = seed
    masks = {'train': [], 'val': [], 'test': []}
    rng = torch.Generator().manual_seed(split_seed)
    masks = {'train': [], 'val': [], 'test': []}
    ds = search_dataset_name(data_name)
    for graph in graphs:
        if ds in ['WebKB', 'WikipediaNet']:
            assert seed in [i for i in range(10)]
            train_mask = graph.train_mask[:, seed]
            val_mask = graph.val_mask[:, seed]
            test_mask = graph.test_mask[:, seed]
            masks['train'].append(train_mask)
            masks['val'].append(val_mask)
            masks['test'].append(test_mask)
        else:
            n_nodes = graph.num_nodes
            split = random_split(range(n_nodes), [0.6, 0.2, 0.2], generator=rng)
            # breakpoint()
            def get_mask(idx):
                mask = torch.zeros(n_nodes, dtype=torch.bool)
                mask[idx] = 1
                return mask
            for i, key in enumerate(masks.keys()):
                masks[key].append( get_mask( split[i] ))
    return masks


# deprecated, ignored it....
# def set_train_val_test_split(
#         seed: int,
#         graphs,
#         data_name: str = None,
#         num_development: int = 1500,
#         num_per_class: int = 20):
#     masks = {'train': [], 'val': [], 'test': []}
#     for graph in graphs:
#         if data_name in _dataset_webkb:
#             assert seed in [i for i in range(10)]
#             train_mask = graph.train_mask[:, seed]
#             val_mask = graph.val_mask[:, seed]
#             test_mask = graph.test_mask[:, seed]
#             # if data_name in _dataset_planetoid:
#             #     train_mask, val_mask, test_mask = train_mask.to(torch.bool), val_mask.to(torch.bool), test_mask.to(torch.bool)
#             masks['train'].append(train_mask)
#             masks['val'].append(val_mask)
#             masks['test'].append(test_mask)
#             # = {'train': train_mask, 'val': val_mask, 'test': test_mask}
#             # data.masks = masks
#         else:
#             rnd_state = np.random.RandomState(seed)
#             num_nodes = graph.y.shape[0]
#             development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
#             test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]
#             train_idx = []
#             rnd_state = np.random.RandomState(seed)
#             for c in range(graph.y.max() + 1):
#                 class_idx = development_idx[np.where(graph.y[development_idx].cpu() == c)[0]]
#                 train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
#             val_idx = [i for i in development_idx if i not in train_idx]
#             def get_mask(idx):
#                 mask = torch.zeros(num_nodes, dtype=torch.bool)
#                 mask[idx] = 1
#                 return mask
#             train_mask, val_mask, test_mask = get_mask(train_idx), get_mask(val_idx),  get_mask(test_idx)
#             masks['train'].append(train_mask)
#             masks['val'].append(val_mask)
#             masks['test'].append(test_mask)
#     return masks

def dataset_statistic(data_names, force_generation = False):
    Infos = dict()
    os.makedirs(os.path.join(DATA_PATH, 'data_stats'), exist_ok = True)
    for name in data_names:
        fname = os.path.join(DATA_PATH, 'data_stats', f'{name}_stats.pkl')
        if force_generation or not os.path.isfile(fname):
            print(f'collecting info of dataset[{name}]')
            ds = data_load(name)
            n_graphs = len(ds) 
            hn = np.mean([ homophily(g.adj_t, g.y, method = 'node') for g in ds])
            he = np.mean([ homophily(g.adj_t, g.y, method = 'edge') for g in ds])
            hei = np.mean([ homophily(g.adj_t, g.y, method = 'edge_insensitive') for g in ds]) 
            n_nodes = sum([g.num_nodes for g in ds])/n_graphs
            n_edges = sum([g.num_edges for g in ds])/n_graphs
            # n_edge_insensitive = sum([g.num_edges for g in ds])/n_graphs
            ds_info = {'Type': search_dataset_name(name), 'N_graphs': n_graphs, 'N_nodes': n_nodes, 'N_edges': n_edges, 'D_feat': ds.num_node_features, 'N_class': ds.num_classes, 'hom_n': hn, 'hom_e': he, 'hom_ei': hei}
            with open(fname, "wb") as f:
                pickle.dump(ds_info, f) 
        else:
            with open(fname, "rb") as f:
                ds_info = pickle.load(f) 
        Infos[name] = ds_info
    Infos = pd.DataFrame.from_dict(Infos, orient = 'index')
    return Infos

if __name__ == "__main__":

    # print( '( ' + ' '.join(_dataset_list) + ' )'  )
    _dataset_list = ['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin', 'Computers', 'Photo', 'Chameleon', 'Squirrel', 'Roman-empire']
    _dataset_list = [ 'Cora'  ]
    # data = data_load('CoauthorCS')
    Infos = dataset_statistic(_dataset_list, True)
    print(Infos)