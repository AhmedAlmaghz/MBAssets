# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : اختبار TTA عمليًّا
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:186
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([keras.Input(shape=(32, 32, 3)),
                          keras.layers.Rescaling(1.0 / 255.0),
                          keras.layers.Conv2D(16, 3, padding="same",
                                              activation="relu"),
                          keras.layers.GlobalAveragePooling2D(),
                          keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy")

# ─── 2. تنبّؤ عاديّ مقابل TTA بأربع نسخ
x_test = np.random.rand(100, 32, 32, 3).astype("float32")
plain = model.predict(x_test, verbose=0)

variants = [x_test,
            keras.ops.flip(x_test, axis=1),
            keras.ops.flip(x_test, axis=2),
            keras.ops.flip(keras.ops.flip(x_test, axis=1), axis=2)]
tta = keras.ops.mean(keras.ops.stack(
    [model(v, training=False) for v in variants], axis=0), axis=0)

diff = float(np.mean(np.abs(np.asarray(plain) - keras.ops.convert_to_numpy(tta))))
print(f"متوسّط الفرق بين التنبّؤين = {diff:.4f}")
# النتيجة المتوقعة: فرق صغير — TTA يثبّت التنبّؤ لا يقلبه.
