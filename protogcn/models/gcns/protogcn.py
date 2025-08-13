# [추론 전용 최종 수정본]
import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from ...utils import Graph
from ..builder import BACKBONES
# gcn.py에서 GCN_Block을 불러오도록 경로 수정
from .utils.gcn import GCN_Block

EPS = 1e-4

# Prototype_Reconstruction_Network는 ProtoGCN 내부에서만 사용되므로 여기에 유지
class Prototype_Reconstruction_Network(nn.Module):
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        query = self.softmax(self.query_matrix(x))
        z = self.memory_matrix(query)
        return self.dropout(z)


@BACKBONES.register_module()
class ProtoGCN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # ... (__init__ 함수는 제공해주신 원본 코드와 동일하게 유지) ...
        graph_cfg = kwargs.get('graph_cfg')
        in_channels = kwargs.get('in_channels', 3)
        base_channels = kwargs.get('base_channels', 96)
        num_stages = kwargs.get('num_stages', 10)
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = kwargs.get('data_bn_type', 'VC')
        self.kwargs = kwargs
        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(kwargs.get('num_person', 2) * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        num_prototype = kwargs.get('num_prototype', 100)
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = kwargs.get('ch_ratio', 2)
        self.inflate_stages = kwargs.get('inflate_stages', [5, 8])
        self.down_stages = kwargs.get('down_stages', [5, 8])
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
        inflate_times = 0
        down_times = 0
        cur_channels = base_channels
        for i in range(2, num_stages + 1):
            stride = 1 + (i in self.down_stages)
            in_channels = cur_channels
            if i in self.inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            cur_channels = out_channels
            down_times += (i in self.down_stages)
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = kwargs.get('pretrained')
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.post = nn.Conv2d(cur_channels, cur_channels, 1)
        self.bn = build_norm_layer(norm_cfg, cur_channels)[1]
        self.relu = nn.ReLU()
        dim = 384
        self.prn = Prototype_Reconstruction_Network(dim, num_prototype)

    def init_weights(self):
        pass # 추론 시에는 불필요

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.reshape((N, M) + x.shape[1:])
        return x