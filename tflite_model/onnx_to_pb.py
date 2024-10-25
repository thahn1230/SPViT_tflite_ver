# 현재 python 3.6으로는 onnx_tf가 돌아가질 않음
# python 3.7이상부터 지원한다고 해서 conda로 새로운 파이썬 버전을 가진 가상환경을 만들어서 해봐야 할듯?
# TFLiteConvert 가상환경으로 만들었음, python=3.8.5

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# ONNX 모델 로드
# onnx_path = "deit_small_patch16_224-cd65a155.onnx" # pre-trained
# onnx_path = "deit-small.onnx" # deit-small
onnx_path = "SPViT30.onnx" # fine-tuning
onnx_model = onnx.load(onnx_path)

# ONNX 모델 유효성 검사
onnx.checker.check_model(onnx_model)

# TensorFlow 모델로 변환
tf_rep = prepare(onnx_model)

# TensorFlow 모델 저장
pb_path = 'SPViT30.pb'
# pb_path = "deit_small_patch16_224-cd65a155.pb"
# pb_path = "deit-small.pb"
tf_rep.export_graph(pb_path)
print(f"TensorFlow 모델이 {pb_path}에 저장되었습니다.")

# # 변환된 TensorFlow 모델 로드 및 확인
# loaded_model = tf.saved_model.load(pb_path)
# print("변환된 TensorFlow 모델 로드 성공")

# # 변환된 모델의 연산 확인
# print("변환된 모델의 연산 확인:")
# for op in loaded_model.signatures['serving_default'].graph.get_operations():
#     print(op.name, op.type)
