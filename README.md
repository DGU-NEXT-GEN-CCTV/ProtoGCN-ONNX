# ProtoGCN-ONNX

이 저장소는 학습된 ProtoGCN 모델을 ONNX 파일로 변환하기 위한 코드를 제공합니다.

### Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. 저장소 복제

```bash
git clone https://github.com/DGU-NEXT-GEN-CCTV/ProtoGCN-ONNX
cd ProtoGCN-ONNX

```

**Step 2**. Conda 가상환경 생성 및 활성화

```bash
conda env create -f protogcn-onnx.yaml
conda activate protogcn-onnx
```

**Step 3**. 라이브러리 설치

```bash
pip install -e .
```

## Preparation

> 변환 과정을 진행하기위해 사전 학습된 ProtoGCN의 설정 및 가중치가 필요합니다.

**설정 파일 경로**: `{repository_root}/configs/{dataset}/{method}.py`

**가중치 파일 경로**: `{repository_root}/checkpoints/{checkpoint}.pth`

## Convert

ONNX 파일을 변환하기 위해 convert_to_onnx.py의 91-93번 줄의 값을 사용자의 환경에 맞게 수정합니다.

```python
    ...(생략)...

# 추천안
91  MODEL_NAME = "protogcn_{dataset}"
92  CONFIG_PATH = 'configs/{dataset}/j.py'
93  CHECKPOINT_PATH = 'checkpoints/{checkpoint}.pth'

...(생략)...
```

ONNX 변환을 위해 아래 명령어를 실행합니다.

```bash
python convert_to_onnx.py
```

변환에 성공했다면 `{repository_root}/onnx_models/{model_name}`이 생성됩니다.

```bash
.
└── onnx_models # ······················· 오닉스 모델 디렉토리
    └── {model_name} # ·················· 변환된 오닉스 모델
        ├── 1 # ························· 버전 디렉토리 (기본값 1)
        │   └── model.onnx # ············ 1 버전 ONNX 모델
        └── config.pbtxt # ·············· ONNX 모델 입/출력 설정 파일
```

## Validate

변환된 ONNX 모델의 유효성을 평가하기 위해 validate_onnx.py의 48번 줄 값을 사용자의 환경에 맞게 수정합니다.

```python
    ...(생략)...

48  MODEL_PATH = 'onnx_models/{model_name}/1/model.onnx'

    ...(생략)...
```
유효성을 평가를 위해 아래 명령어를 실행합니다.
```bash
python validate_onnx.py
```

유효성 평가는 2단계로 진행되며 첫번째는 CPU 세션에서 추론, 두번째는 GPU 세션에서 추론으로 구성되어 있습니다.

다만, 이 지표는 CUDA 등 환경 설정으로 인해 모델에 문제가 없음에도 오류가 발생할 수 있습니다.

> Note. 모델 변환에 심각한 문제가 발생하지 않았다면, 일반적으로 CPU 세션에서 정상적으로 동작합니다.

## ETC.

