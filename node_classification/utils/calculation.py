import torch
import torch.nn.functional as F
from torch_geometric.utils import to_edge_index
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


# def calculate_homophily(x1, x2, het_mode = None):
#     assert x1.shape == x2.shape
    
#     if het_mode is None or het_mode=='original':
#         out = torch.ones(x1.shape[:-1]).to(device = x1.device)
#     elif het_mode == 'homophily':
#         out = F.cosine_similarity(x1, x2, dim=-1)
#     elif het_mode == 'heterophily':
#         out = 1.0 - F.cosine_similarity(x1, x2, dim=-1)
#     else:
#         raise NameError(f'The het_mode {het_mode} value incorrectly')
#     print('ss calculated!')
#     return out
def calculate_homophily(x, edge_index, het_mode = None):

    # if edge_index.layout == torch.sparse_csr:
    #     idx1, idx2 = edge_index.to_sparse().indices()
    if isinstance(edge_index, SparseTensor) or edge_index.layout == torch.sparse_csr:
        edge_index = to_edge_index(edge_index)[0]
    x1, x2 = x[edge_index[0]], x[edge_index[1]]
    if het_mode is None or het_mode=='original':
        out = torch.ones(x1.shape[:-1]).to(device = x1.device)
    elif het_mode == 'homophily':
        out = F.cosine_similarity(x1, x2, dim=-1)
    elif het_mode == 'heterophily':
        out = 1.0 - F.cosine_similarity(x1, x2, dim=-1)
    else:
        raise NameError(f'The het_mode {het_mode} value incorrectly')
    # print('ss calculated!')
    return out

    
def MAD(edge_index, hx):
    # d_src, d_dst = degree[edge_index[:,0]], degree[edge_index[:, 1]]
    h_src, h_dst = hx[edge_index[:,0]], hx[edge_index[:, 1]]
    mad = torch.mean(F.cosine_similarity(h_src, h_dst, dim=-1))
    return mad

def dirichilet_energy(edge_index, hx, degree):
    d_src, d_dst = degree[edge_index[:,0]], degree[edge_index[:, 1]]
    h_src, h_dst = hx[edge_index[:,0]], hx[edge_index[:, 1]]
    tmp = torch.norm(h_src/(d_src+1)**0.5 - h_dst/(d_dst+1)**0.5, dim = -1)
    energy = torch.norm(tmp)
    return energy