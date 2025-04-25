from .gnns import GCN, GAT, GIN, GraphSAGE
# from .layers import _single_modes, _mix_modes, _all_modes
from torch import nn
import torch

# * het modes
_single_modes = ['original', 'heterophily', 'homophily']
_mix_modes = ['mix', 'mixfake']
_all_modes = _single_modes + _mix_modes

# * model types (gnn types)
_dict_gnnclass = {'GCN': GCN, 'GAT': GAT, 'GIN': GIN, 'GraphSAGE': GraphSAGE}
_model_types = list(_dict_gnnclass.keys() )

jk_mode =  None
# norm_mode = 'batch_norm'
# norm_mode = None
class MixGNN(nn.Module):
    def __init__(self, gnn_type, in_channels,hidden_channels, out_channels, num_layers, het_mode = 'mix', dropout=0.1, norm = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels =  out_channels
        self.hidden_channels = hidden_channels
        self.het_mode = het_mode
        GNN = _dict_gnnclass[gnn_type]
        if het_mode == 'mix':
            het_modes = ['original', 'heterophily', 'homophily']
        elif het_mode == 'mixfake':
            het_modes = ['original', 'original', 'original']
        else:
            het_modes = [het_mode]
        self.subgnns = nn.ModuleList([
            GNN(in_channels = in_channels,
            hidden_channels = hidden_channels,
            out_channels = out_channels,
            num_layers= num_layers,
            het_mode = het_mode, jk=jk_mode, norm = norm, dropout = dropout) for het_mode in het_modes])
        
        
        self.lin = nn.Linear(self.out_channels * 3, self.out_channels, bias=True)
    def forward(self,  x, edge_index, **kwargs):
        outs = [gnn(x, edge_index, **kwargs) for gnn in self.subgnns ]
        out = self.lin(torch.cat(outs, dim=-1))
        return out





def model_initialization(gnn_type, in_channels, hidden_channels, out_channels,num_layers, het_mode = None,  dropout=0.1, norm_mode = None):
    
    # todo if the normalization need to be changed?
    # the activation func is relu as default
    # normalization is Identity as default
    assert gnn_type in _model_types, f'`{gnn_type}` not in {_model_types}'
    assert het_mode in _all_modes+[None], f'`{het_mode}` not in {_all_modes +[None]}'
    model = None
    if het_mode in _single_modes:
        GNN = _dict_gnnclass[gnn_type]
        model = GNN(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            out_channels = out_channels,
            num_layers= num_layers,
            het_mode = het_mode, jk=jk_mode, norm = norm_mode, dropout = dropout)
    elif het_mode in _mix_modes:
        model = MixGNN(
            gnn_type = gnn_type,
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            out_channels = out_channels,
            num_layers= num_layers,
            het_mode = het_mode, dropout = dropout, norm = norm_mode)
    return model