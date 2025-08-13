import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import math
import numpy as np
import torch.nn.functional as F

from ..builder import HEADS
from .base import *


@HEADS.register_module()
class SimpleHead(BaseHead): # BaseHead가 정의되어 있다고 가정합니다.

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout=0.,
                 init_std=0.01,
                 **kwargs): # loss_cls 등 훈련 전용 인수는 **kwargs로 받음
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        
        # Dropout은 model.eval()에서 적용되지 않지만, 명시적으로 정의
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else None

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)
        
        # forward에서 동적으로 생성하지 않도록 __init__에서 미리 정의
        self.pool = nn.AdaptiveAvgPool2d(1)

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # GCN 백본의 출력은 5D 텐서 (N, M, C, T, V)
        # if문 없이 GCN 출력을 처리하는 단일 경로만 남김
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = self.pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        # Dropout은 model.eval() 모드에서 자동으로 비활성화됨
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score