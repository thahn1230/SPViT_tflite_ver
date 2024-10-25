# import torch
# import torch.onnx
# from transformers import ViTForImageClassification
# import os
# import sys

# # huggingface에서 지원하는 transformers 모듈에 있는 deit small model 가져옴
# model = ViTForImageClassification.from_pretrained('facebook/deit-small-patch16-224')

# # 평가 모드 전환
# model.eval()

# # 모델을 GPU로 이동
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)

# # 더미 입력 텐서 생성 및 GPU로 이동
# # deit small이 224x224
# dummy_input = torch.randn(1, 3, 224, 224).to(device)

# # 모델을 ONNX 형식으로 변환 및 저장
# onnx_path = "deit-small.onnx"
# torch.onnx.export(
#     model,                  # 변환할 모델
#     dummy_input,            # 예제 입력 데이터
#     onnx_path,              # ONNX 파일 저장 경로
#     export_params=True,     # 모델의 학습된 파라미터 저장
#     opset_version=14,       # ONNX opset 버전
#     input_names=['input'],  # 입력 텐서 이름
#     output_names=['output'],# 출력 텐서 이름
#     # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # 동적 축 정의
# )
# print(f"ONNX 모델이 {onnx_path}에 저장되었습니다.")


import torch
import torch.onnx
import os
import sys

# Append parent directory to system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vit import VisionTransformerDiffPruning
from vit_l2 import checkpoint_filter_fn, _cfg

# Ensure these are tensors or Python primitives, not numpy types
KEEP_RATE = torch.tensor([0.617, 0.369, 0.137], dtype=torch.float)
PRUNING_LOC = torch.tensor([3, 6, 9], dtype=torch.long)

model = VisionTransformerDiffPruning(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE
)

# model_path = '../deit_small_patch16_224-cd65a155.pth'
model_path = '../checkpoint_best30.pth'
checkpoint = torch.load(model_path, map_location="cpu")
ckpt = checkpoint_filter_fn(checkpoint, model)
model.default_cfg = _cfg()
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(device)
model.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)

onnx_path = "SPViT30.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print(f"ONNX model has been saved to {onnx_path}.")
