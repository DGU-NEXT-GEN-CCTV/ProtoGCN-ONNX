# [추론 전용 최종 수정본]
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

# tcn.py 파일에서 mstcn과 unit_tcn을 불러옵니다.
from .tcn import mstcn, unit_tcn

print("✅ [검증] 최신 gcn.py 파일이 로드되었습니다!")

# ==============================================================================
# unit_gcn (FINAL INFERENCE VERSION)
# ==============================================================================
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ratio = kwargs.get('ratio', 0.125)
        mid_channels = int(self.ratio * out_channels)
        self.mid_channels = mid_channels
        norm = kwargs.get('norm', 'BN')
        act = kwargs.get('act', 'ReLU')
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)
        self.intra_act = kwargs.get('intra_act', 'softmax')
        self.inter_act = kwargs.get('inter_act', 'tanh')
        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)
        A = self.A
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        tmp_x = x
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        if self.inter_act == 'tanh':
            inter_graph = self.tanh(diff)
        elif self.inter_act == 'softmax':
            inter_graph = self.softmax(diff)
        else:
            inter_graph = diff
        inter_graph = inter_graph * self.alpha[0]
        A = inter_graph + A
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        if self.intra_act == 'softmax':
            intra_graph = self.softmax(intra_graph)
        elif self.intra_act == 'sigmoid':
            intra_graph = self.sigmoid(intra_graph)
        else:
            intra_graph = intra_graph
        intra_graph = intra_graph * self.beta[0]
        A = intra_graph + A
        A = A.squeeze(3)
        x = torch.matmul(pre_x, A).contiguous()
        x = x.reshape(n, -1, t, v).contiguous()
        x = self.post(x)
        return self.act(self.bn(x) + res)

# ==============================================================================
# GCN_Block (FINAL INFERENCE VERSION)
# ==============================================================================
class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith('gcn_')}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith('tcn_')}
        remaining_kwargs = {k: v for k, v in kwargs.items() if not (k.startswith('tcn_') or k.startswith('gcn_'))}
        assert len(remaining_kwargs) == 0, f"Unexpected kwargs found in GCN_Block: {remaining_kwargs.keys()}"
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, **tcn_kwargs, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x)