import torch
from protogcn.models.gcns.protogcn import GCN_Block
from protogcn.utils import Graph

def test_gcn_block_conversion():
    print("--- GCN_Block 고립 테스트 시작 ---")
    
    # 1. GCN_Block 인스턴스 생성에 필요한 설정값 정의
    in_channels = 3
    out_channels = 96
    
    graph_cfg = dict(layout='coco_new', mode='random', num_filter=8, init_off=.04, init_std=.02)
    try:
        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        print("✅ 그래프(A) 행렬 생성 성공")
    except Exception as e:
        print(f"❌ 그래프(A) 행렬 생성 실패: {e}")
        print("   'protogcn/utils/graph.py' 파일과 'layout' 설정이 올바른지 확인하세요.")
        return

    # 2. GCN_Block에 전달할 kwargs 딕셔너리를 명시적으로 생성합니다.
    #    이렇게 하면 코드의 의도가 명확해지고 실수를 줄일 수 있습니다.
    block_kwargs = {
        'act': 'ReLU',
        'norm': 'BN'
    }
    print(f"GCN_Block에 전달될 kwargs: {block_kwargs}")

    # 3. GCN_Block 단일 모듈 인스턴스화
    try:
        # stride와 residual은 기본값이 있으므로 kwargs로 전달하지 않아도 됩니다.
        gcn_block = GCN_Block(in_channels=in_channels, 
                              out_channels=out_channels, 
                              A=A, 
                              stride=1, 
                              residual=False, 
                              **block_kwargs)
        gcn_block.eval()
        print("✅ GCN_Block 모듈 생성 성공")
    except TypeError as e:
        print(f"❌ GCN_Block 모듈 생성 실패: {e}")
        print("\n[중요] 이 오류는 'protogcn/models/gcns/utils.py' 파일의 GCN_Block.__init__ 함수가 `**kwargs`를 받도록 정의되지 않았음을 의미합니다.")
        print("파일이 최신 버전으로 올바르게 저장되었는지 다시 한번 확인해주세요.")
        return
    except Exception as e:
        print(f"❌ GCN_Block 모듈 생성 중 알 수 없는 오류 발생: {e}")
        return

    # 4. 더미 입력 데이터 생성 (N*M, C, T, V)
    dummy_input = torch.randn(1, in_channels, 100, 20)
    print(f"더미 입력 Shape: {dummy_input.shape}")

    # 5. 고립된 모듈을 ONNX로 변환
    onnx_file_path = "gcn_block_isolated.onnx"
    try:
        # 추론 전용으로 수정한 forward는 입력으로 x 하나만 받습니다.
        torch.onnx.export(
            gcn_block,
            dummy_input,
            onnx_file_path,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        print(f"✅ ONNX 변환 성공: {onnx_file_path}")
        print("\n이제 'gcn_block_isolated.onnx' 파일을 GPU에서 실행하여 오류가 재현되는지 확인하세요.")
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return

if __name__ == '__main__':
    test_gcn_block_conversion()