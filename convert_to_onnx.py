import numpy as np
import torch
import torch.onnx
from mmcv import Config
from mmcv.runner import load_checkpoint
from protogcn.models import build_model
import shutil

config_file = 'configs/selfharm/j.py'
checkpoint_file = 'checkpoints/protogcn_pytorch_selfharm.pth'

# 1. 원본 PyTorch 모델과 가중치 로드
cfg = Config.fromfile(config_file)
model = build_model(cfg.model)
load_checkpoint(model, checkpoint_file)

model.eval()

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        return self.model.forward_inference(keypoint, **kwargs)

wrapped_model = Wrapper(model)


# 2. 모델에 들어갈 더미 입력(dummy input)을 생성
# -------------------------
# BS, NC, T, V, C
# BS: 배치 사이즈(사람 수)
# NC: 클립 수(기본값 1)
# T: 시퀀스 길이(프레임 수, 기본값 100)
# V: 키포인트 수(기본값 17)
# C: 채널 수(x,y,confidence)
# -------------------------
# dummy_input = torch.randn(1, 1, 100, 20, 3)
dummy_input = torch.tensor(np.load('data/selfharm/test_keypoint.npy'))

# 3. ONNX로 변환합니다.
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 4. ONNX 모델을 triton 서버로 복사
shutil.copy("model.onnx", "/workspace/hyunsu/project_cctv/triton/triton-server/onnx_model/protogcn_selfharm/1/model.onnx")

print("✅ ONNX 모델 변환이 완료되었습니다.")