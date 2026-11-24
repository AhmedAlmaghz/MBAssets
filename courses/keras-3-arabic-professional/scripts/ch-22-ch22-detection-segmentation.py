#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الكشف والتجزئة الدلالية
Keras 3 in Arabic — Detection Segmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-22-ch22-detection-segmentation (lesson 22)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-22-ch22-detection-segmentation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الكشف والتجزئة الدلالية
Keras 3 in Arabic — Detection Segmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-22-ch22-detection-segmentation (lesson 22)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-22-ch22-detection-segmentation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الكشف والتجزئة الدلالية
Keras 3 in Arabic — Detection Segmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-22-ch22-detection-segmentation (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch22_detection_segmentation/001_s22_5_keras_detection_head_coordinates.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch22_detection_segmentation/002_s22_6_net_segmentation_model_mini.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch22_detection_segmentation/003_iou_dice_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب IoU وDice يدويًّا
# المصدر  : src/chapters/ch22_detection_segmentation.md:173
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch22_detection_segmentation/004_map_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب mAP يدويًّا لمربّعات محيط
# المصدر  : src/chapters/ch22_detection_segmentation.md:206
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب IoU وDice يدويًّا
# المصدر  : src/chapters/ch22_detection_segmentation.md:173
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب mAP يدويًّا لمربّعات محيط
# المصدر  : src/chapters/ch22_detection_segmentation.md:206
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الكشف والتجزئة الدلالية
Keras 3 in Arabic — Detection Segmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-22-ch22-detection-segmentation (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch22_detection_segmentation/001_s22_5_keras_detection_head_coordinates.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch22_detection_segmentation/002_s22_6_net_segmentation_model_mini.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch22_detection_segmentation/003_iou_dice_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب IoU وDice يدويًّا
# المصدر  : src/chapters/ch22_detection_segmentation.md:173
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch22_detection_segmentation/004_map_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب mAP يدويًّا لمربّعات محيط
# المصدر  : src/chapters/ch22_detection_segmentation.md:206
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch22_detection_segmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب IoU وDice يدويًّا
# المصدر  : src/chapters/ch22_detection_segmentation.md:173
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب mAP يدويًّا لمربّعات محيط
# المصدر  : src/chapters/ch22_detection_segmentation.md:206
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-22-ch22-detection-segmentation.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).
