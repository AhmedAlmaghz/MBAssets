# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.5 التنفيذ في Keras
# المصدر  : src/chapters/ch24_super_resolution.md:60
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.
