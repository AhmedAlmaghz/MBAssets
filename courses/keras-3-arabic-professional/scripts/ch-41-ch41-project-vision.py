#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تصنيف صور إنتاجيّ
Keras 3 in Arabic — Project Vision

Book slug : keras-3-arabic-professional
Chapter   : ch-41-ch41-project-vision (lesson 41)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-41-ch41-project-vision.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تصنيف صور إنتاجيّ
Keras 3 in Arabic — Project Vision

Book slug : keras-3-arabic-professional
Chapter   : ch-41-ch41-project-vision (lesson 41)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-41-ch41-project-vision.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تصنيف صور إنتاجيّ
Keras 3 in Arabic — Project Vision

Book slug : keras-3-arabic-professional
Chapter   : ch-41-ch41-project-vision (lesson 41)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch41_project_vision/001_s41_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.2 البيانات
# المصدر  : src/chapters/ch41_project_vision.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch41_project_vision/002_s41_3_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.3 النموذج
# المصدر  : src/chapters/ch41_project_vision.md:56
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch41_project_vision/003_s41_4_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.4 التدريب
# المصدر  : src/chapters/ch41_project_vision.md:90
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch41_project_vision/004_s41_5_evaluation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.5 التقييم الشامل
# المصدر  : src/chapters/ch41_project_vision.md:114
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch41_project_vision/005_s41_6_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.6 التصدير والتكميم
# المصدر  : src/chapters/ch41_project_vision.md:134
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch41_project_vision/006_s41_7_serving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.7 الخدمة
# المصدر  : src/chapters/ch41_project_vision.md:145
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch41_project_vision/007_s41_9_pipeline_data_efficient_production.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.9 خطّ البيانات الفعّال في الإنتاج
# المصدر  : src/chapters/ch41_project_vision.md:183
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch41_project_vision/008_s41_10_calibration_probabilities.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.10 معايرة الاحتمالات
# المصدر  : src/chapters/ch41_project_vision.md:221
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.2 البيانات
# المصدر  : src/chapters/ch41_project_vision.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.3 النموذج
# المصدر  : src/chapters/ch41_project_vision.md:56
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.4 التدريب
# المصدر  : src/chapters/ch41_project_vision.md:90
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.5 التقييم الشامل
# المصدر  : src/chapters/ch41_project_vision.md:114
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.6 التصدير والتكميم
# المصدر  : src/chapters/ch41_project_vision.md:134
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.7 الخدمة
# المصدر  : src/chapters/ch41_project_vision.md:145
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.9 خطّ البيانات الفعّال في الإنتاج
# المصدر  : src/chapters/ch41_project_vision.md:183
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.10 معايرة الاحتمالات
# المصدر  : src/chapters/ch41_project_vision.md:221
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تصنيف صور إنتاجيّ
Keras 3 in Arabic — Project Vision

Book slug : keras-3-arabic-professional
Chapter   : ch-41-ch41-project-vision (lesson 41)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch41_project_vision/001_s41_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.2 البيانات
# المصدر  : src/chapters/ch41_project_vision.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch41_project_vision/002_s41_3_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.3 النموذج
# المصدر  : src/chapters/ch41_project_vision.md:56
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch41_project_vision/003_s41_4_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.4 التدريب
# المصدر  : src/chapters/ch41_project_vision.md:90
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch41_project_vision/004_s41_5_evaluation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.5 التقييم الشامل
# المصدر  : src/chapters/ch41_project_vision.md:114
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch41_project_vision/005_s41_6_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.6 التصدير والتكميم
# المصدر  : src/chapters/ch41_project_vision.md:134
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch41_project_vision/006_s41_7_serving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.7 الخدمة
# المصدر  : src/chapters/ch41_project_vision.md:145
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch41_project_vision/007_s41_9_pipeline_data_efficient_production.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.9 خطّ البيانات الفعّال في الإنتاج
# المصدر  : src/chapters/ch41_project_vision.md:183
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch41_project_vision/008_s41_10_calibration_probabilities.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.10 معايرة الاحتمالات
# المصدر  : src/chapters/ch41_project_vision.md:221
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch41_project_vision.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.2 البيانات
# المصدر  : src/chapters/ch41_project_vision.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.3 النموذج
# المصدر  : src/chapters/ch41_project_vision.md:56
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.4 التدريب
# المصدر  : src/chapters/ch41_project_vision.md:90
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.5 التقييم الشامل
# المصدر  : src/chapters/ch41_project_vision.md:114
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.6 التصدير والتكميم
# المصدر  : src/chapters/ch41_project_vision.md:134
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.7 الخدمة
# المصدر  : src/chapters/ch41_project_vision.md:145
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.9 خطّ البيانات الفعّال في الإنتاج
# المصدر  : src/chapters/ch41_project_vision.md:183
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.10 معايرة الاحتمالات
# المصدر  : src/chapters/ch41_project_vision.md:221
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("final_model.keras")

# ─── 2. تصدير إلى LiteRT (في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-41-ch41-project-vision.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.
