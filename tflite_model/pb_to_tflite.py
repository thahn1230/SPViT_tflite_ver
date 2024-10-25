import tensorflow as tf

# 모델 로드
# saved_model_dir = "deit_small_patch16_224-cd65a155.pb"
# saved_model_dir = "deit-small.pb"
saved_model_dir = "SPViT30.pb"
model = tf.saved_model.load(saved_model_dir)

# 컨버터 로드
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# unsupported operations을 위해서 만듦
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS  # Enable TensorFlow Lite ops.
    # tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops.
]

# optimizations options, 선택적이긴 한데 일단은 default로 해놓음
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 모델 convert
tflite_model = converter.convert()

# .tflite파일로 변환
# tflite_model_path = "deit_small_patch16_224-cd65a155.tflite"
tflite_model_path = "SPViT30.tflite"
# tflite_model_path = "deit-small.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite 모델이 {tflite_model_path}에 저장되었습니다.")