import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.125,
                 intra_act='softmax',
                 inter_act='tanh',
                 norm='BN',
                 act='ReLU',
                 **kwargs): # kwargs를 받아들이도록 추가
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)
        self.intra_act = intra_act
        self.inter_act = inter_act

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
        # getattr 대신 명시적인 if/else 사용
        if self.inter_act == 'tanh':
            inter_graph = self.tanh(diff)
        elif self.inter_act == 'softmax':
            inter_graph = self.softmax(diff)
        else:
            inter_graph = diff
            
        inter_graph = inter_graph * self.alpha[0]
        A = inter_graph + A
        
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        # getattr 대신 명시적인 if/else 사용
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
        
        # 최종 출력 (단일 텐서만 반환)
        return self.act(self.bn(x) + res)