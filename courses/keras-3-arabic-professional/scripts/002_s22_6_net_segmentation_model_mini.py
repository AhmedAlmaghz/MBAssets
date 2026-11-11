# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : 22.6 التجزئة: نموذج U-Net مصغّر
# المصدر  : src/chapters/ch22_detection_segmentation.md:109
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: 300 صورة 64x64 مع قناع ثنائي
n = 300
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y_mask = (np.random.rand(n, 64, 64, 1) < 0.3).astype("float32")

# ─── 2. U-Net مصغّرة
def conv_block(x, filters):
    """كتلة تجزئة قياسية: التفافان مع التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

inp = keras.Input(shape=(64, 64, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)

# encoder
c1 = conv_block(h, 16)                       # 64
p1 = keras.layers.MaxPooling2D(2)(c1)        # 32
c2 = conv_block(p1, 32)                      # 32
p2 = keras.layers.MaxPooling2D(2)(c2)        # 16
c3 = conv_block(p2, 64)                      # 16

# decoder مع اتصالات تخطّي
u2 = keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c3)   # 32
m2 = keras.layers.Concatenate()([u2, c2])
c4 = conv_block(m2, 32)
u1 = keras.layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c4)   # 64
m1 = keras.layers.Concatenate()([u1, c1])
c5 = conv_block(m1, 16)

out = keras.layers.Conv2D(1, 1, activation="sigmoid")(c5)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.MeanIoU(num_classes=2)])
model.fit(x, y_mask, epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MeanIoU ≈ 0.55 بعد 10 دورات على بيانات عشوائية.
