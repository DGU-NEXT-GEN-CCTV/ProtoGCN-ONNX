import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import unit_gcn, mstcn, unit_tcn

EPS = 1e-4


class GCN_Block(nn.Module):
    """
    GCN Block for ST-GCN models.
    
    [추론 전용으로 수정된 최종 버전]
    - __init__이 **kwargs를 받도록 수정되었습니다.
    - forward 함수는 단일 텐서만 입출력합니다.
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        
        # 'act', 'norm'과 같은 공통 인자를 gcn과 tcn 양쪽에 전달하기 위한 로직
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        
        # 각 모듈에 맞는 kwargs만 필터링
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith('gcn_')}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith('tcn_')}
        
        # 남은 kwargs가 없는지 확인
        remaining_kwargs = {k: v for k, v in kwargs.items() if not (k.startswith('tcn_') or k.startswith('gcn_'))}
        assert len(remaining_kwargs) == 0, f"Unexpected kwargs found in GCN_Block: {remaining_kwargs.keys()}"

        # 핵심 모듈 정의
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        # Residual connection 설정
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        [추론 전용 forward]
        - 입력: x (텐서)
        - 출력: relu(x) (텐서)
        """
        res = self.residual(x)
        
        # unit_gcn이 단일 텐서만 반환하므로, x만 받도록 수정
        x = self.gcn(x)
        
        x = self.tcn(x) + res
        
        # 최종적으로 단일 텐서만 반환
        return self.relu(x)


"""
****************************************
*** Prototype Reconstruction Network ***
****************************************
"""  
class Prototype_Reconstruction_Network(nn.Module):
    
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias=False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1) # torch.softmax 대신 nn.Softmax 사용
        # Dropout은 model.eval()에서 자동으로 비활성화되므로 그대로 두거나,
        # 추론 전용 코드에서는 삭제해도 무방합니다.
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        query = self.softmax(self.query_matrix(x))
        z = self.memory_matrix(query)
        # Dropout은 model.eval() 모드에서 적용되지 않습니다.
        return self.dropout(z)


@BACKBONES.register_module()
class ProtoGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 # ... (나머지 __init__ 인수는 기존과 동일) ...
                 **kwargs):
        super().__init__()
        # ... (기존 __init__ 코드는 그대로 사용) ...
        # 이 부분은 수정할 필요 없습니다.
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = kwargs.get('data_bn_type', 'VC') # 기본값 설정
        self.kwargs = kwargs

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(kwargs.get('num_person', 2) * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        num_prototype = kwargs.pop('num_prototype', 100)
        num_stages = kwargs.get('num_stages', 10)
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
        self.pretrained = kwargs.get('pretrained', None)
        
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        
        self.post = nn.Conv2d(cur_channels, cur_channels, 1)
        self.bn = build_norm_layer(norm_cfg, cur_channels)[1]
        self.relu = nn.ReLU()
        
        dim = 384
        self.prn = Prototype_Reconstruction_Network(dim, num_prototype)

    def init_weights(self):
        # ... (기존과 동일) ...
        pass

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # GCN 블록을 순차적으로 통과시킵니다.
        # GCN_Block이 이제 단일 텐서만 반환하므로, x만 받습니다.
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        
        # 최종 특징(feature) 텐서의 shape을 원복합니다.
        x = x.reshape((N, M) + x.shape[1:])
        
        # 'get_graph' 및 'reconstructed_graph' 관련 로직은
        # 추론에 필요 없으므로 모두 삭제되었습니다.
        
        # 최종 특징(feature) 텐서 'x'만 반환합니다.
        return x