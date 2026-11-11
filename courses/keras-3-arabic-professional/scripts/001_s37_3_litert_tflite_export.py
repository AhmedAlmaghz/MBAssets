# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.3 التصدير إلى LiteRT/TFLite
# المصدر  : src/chapters/ch37_deployment.md:40
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.
