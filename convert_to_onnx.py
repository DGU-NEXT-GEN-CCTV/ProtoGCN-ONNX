import os
import numpy as np
import torch
import torch.onnx
from mmcv import Config
from mmcv.runner import load_checkpoint
from protogcn.models import build_model

def gen_onnx_config(model_name: str, model_dir: str):
    config_content = f"""name: "{model_name}"
    platform: "onnxruntime"
    max_batch_size: 0
    input [
      {{
        name: "input"
        data_type: TYPE_FP32
        dims: [ -1, 1, 100, 20, 3 ]
      }}
    ]
    output [
      {{
        name: "output"
        data_type: TYPE_FP32
        dims: [ -1, 2 ]
      }}
    ]"""
        
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✅ config.pbtxt 파일이 생성되었습니다: {config_path}")

def convert(model_name: str, config_path: str, checkpoint_path: str):
    model_dir = os.path.join("onnx_models", model_name)
    model_version_dir = os.path.join(model_dir, "1")
    model_output_path = os.path.join(model_version_dir, "model.onnx")
    os.makedirs("onnx_models", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_version_dir, exist_ok=True)

    # 1. 원본 PyTorch 모델과 가중치 로드
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path)

    # 모델을 평가 모드로 설정
    model.eval()

    # 모델의 forward 메서드를 래핑합니다.
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
    dummy_input = torch.randn(1, 1, 100, 20, 3)

    # 3. ONNX로 변환합니다.
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        model_output_path,
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
    print(f"✅ ONNX 모델 변환이 완료되었습니다. 저장 경로: {model_dir}")
    
    gen_onnx_config(model_name, model_dir)

if __name__ == "__main__":
    MODEL_NAME = 'protogcn_selfharm'
    CONFIG_PATH = 'configs/selfharm/j.py'
    CHECKPOINT_PATH = 'checkpoints/protogcn_pytorch_selfharm.pth'
    convert(MODEL_NAME, CONFIG_PATH, CHECKPOINT_PATH)