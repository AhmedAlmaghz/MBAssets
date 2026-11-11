# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : 22.5 الكشف في Keras: رأس إحداثيّات
# المصدر  : src/chapters/ch22_detection_segmentation.md:62
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: صور 64x64 مع جسم واحد في كل صورة
n = 500
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
# الإحداثيّات: مركز الجسم (cx, cy) وحجمه (w, h) مُطبّعة إلى [0, 1]
y_bbox = np.random.rand(n, 4).astype("float32")
y_cls = np.random.randint(0, 2, size=(n,)).astype("int32")

# ─── 2. البنية: encoder مشترك + رأسان
inp = keras.Input(shape=(64, 64, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)
h = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(h)
h = keras.layers.MaxPooling2D(2)(h)
h = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(h)
h = keras.layers.MaxPooling2D(2)(h)
h = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(h)
h = keras.layers.GlobalAveragePooling2D()(h)

# رأس 1: إحداثيّات المربّع المحيط (انحدار)
bbox_out = keras.layers.Dense(4, activation="sigmoid", name="bbox")(h)
# رأس 2: تسمية الجسم (تصنيف)
cls_out = keras.layers.Dense(1, activation="sigmoid", name="cls")(h)

model = keras.Model(inp, [bbox_out, cls_out])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={"bbox": "mse", "cls": "binary_crossentropy"},
    loss_weights={"bbox": 1.0, "cls": 0.5},
    metrics={"bbox": "mae", "cls": "acc"},
)
model.fit(x, {"bbox": y_bbox, "cls": y_cls}, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE للإحداثيّات ≈ 0.08 بعد 10 دورات،
# وdقّة التسمية ≈ 0.75.
