import torch
from torch.nn import Linear
from models.layers import gat_conv, gcn_conv, gin_conv, sage_conv
import torch.nn as nn
_single_modes = ['original', 'heterophily', 'homophily']
_mix_modes = ['mix', 'mixfake']
_all_modes = _single_modes + _mix_modes

# _conv_dict = {'GCN': gcn_conv, "GAT": gat_conv, "GIN": gin_conv, "GraphSAGE": sage_conv}


class HetGNNConv(torch.nn.Module):
    def __init__(self, convs, in_channels, out_channels, het_mode = None):
        super().__init__()
        '''
        convs: convolutional layers
            - one nn, when `het_mode` in `_single_modes`
            - list of 3 nns, when `het_mode` in `_mixed_modes`
        
        '''
        self.in_channels = in_channels
        self.out_channels =  out_channels
        self.het_mode = het_mode

        assert self.het_mode in _all_modes+[None] # mode not existed, mode is None, or mode is 'orignial' are the same
            
        if het_mode in _single_modes + [None]:
            self.gnn_conv = convs
        elif het_mode in _mix_modes:
            self.gnn_convs = convs
            self.lin = Linear(self.out_channels * 3, self.out_channels, bias=False,)
        
    def forward(self, x, edge_index, **kwargs):
        out = None
        assert self.het_mode in _all_modes + [None] 
        
        if self.het_mode in [None] + _single_modes:
            out = self.gnn_conv(x,edge_index, het_mode = self.het_mode, **kwargs)
        elif self.het_mode in _mix_modes:
            if self.het_mode == 'mix':
                modes = ['original', 'heterophily', 'homophily']
            elif self.het_mode == 'mixfake':
                modes = ['original', 'original', 'original']
            outs = [conv(x,edge_index, het_mode = m, **kwargs) 
                    for m, conv in zip(modes, self.gnn_convs ) ]
            out = self.lin(torch.cat(outs, dim=-1))
        return out
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn_conv.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
            
    def switch_het_mode(self, het_mode):
        self.het_mode = het_mode
        if het_mode in ['mix', 'mixfake'] and not hasattr(self, 'lin'):
            self.lin = Linear(self.out_channels * 3, self.out_channels, bias=False,
                              weight_initializer='glorot')