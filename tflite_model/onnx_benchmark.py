import onnxruntime as ort
import numpy as np
import time

# ONNX 모델 로드
# model_path = 'deit_small_patch16_224-cd65a155.onnx'
# model_path = 'deit-small.onnx'
model_path = 'SPViT30.onnx'
session = ort.InferenceSession(model_path)

# 입력 텐서 준비 (모두 0으로 채움)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_data = np.zeros(input_shape, dtype=np.float32)

# 지연 시간 측정 루프
inference_times = []

print(f'{model_path}')
for _ in range(10):  
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    end_time = time.time()
    
    inference_times = (end_time - start_time)
    print(f'Average Inference Time: {inference_times:.6f} seconds')

# 성능 측정 결과 출력
# average_inference_time = sum(inference_times) / len(inference_times)

# print(f'Average Inference Time: {average_inference_time:.6f} seconds')
# 