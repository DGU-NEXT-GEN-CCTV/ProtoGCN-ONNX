import onnxruntime as ort
import numpy as np
def valid_onnx_model(provider, model_path):
    so = ort.SessionOptions()
    so.log_severity_level = 0
    
    unit = 'CPU' if 'CPU' in provider else 'GPU'
    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        cur_provider = session.get_providers()
        if provider not in cur_provider:
            print(f"❌ {unit} provider is not available.")
            return None
        print(f"✅ ONNX session created successfully on {unit}.")
    except Exception as e:
        print(f"❌ Failed to create ONNX session: {e}")
        # 여기서 오류가 난다면 모델 또는 환경 문제
        exit()
        
    # 모델의 입력 정보 확인
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    print(f"Input name: {input_name}, shape: {input_shape}")
    print(f"Output name: {output_name}, shape: {output_shape}")

    # shape에 맞는 더미 데이터 생성 (None으로 된 차원은 1로 가정)
    dummy_input = np.random.rand(1, 1, 100, 20, 3).astype(np.float32)

    print(f"Running inference with input shape: {dummy_input.shape}, output shape: {output_shape}")

    try:
        # 추론 실행
        outputs = session.run(None, {input_name: dummy_input})
        print(f"✅ ONNX model inference successful on {unit}!")
        print(outputs)
        for i, out in enumerate(outputs):
            print(f"  Output {i} shape: {out.shape}")
    except Exception as e:
        # 이 스크립트 실행 시 동일한 오류가 발생하면 모델 자체의 문제입니다.
        print(f"❌ ONNX model inference failed: {e}")
    print('-'*50)

if __name__ == "__main__":
    # CPU와 GPU에서 각각 ONNX 모델을 검증합니다.
    MODEL_PATH = 'onnx_models/protogcn_selfharm/1/model.onnx'
    
    # CPU에서 모델 검증
    valid_onnx_model("CPUExecutionProvider", MODEL_PATH)

    # GPU에서 모델 검증
    valid_onnx_model("CUDAExecutionProvider", MODEL_PATH)