# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.5 التصدير إلى ONNX
# المصدر  : src/chapters/ch37_deployment.md:77
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.
