#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكة CNN من الصفر
Keras 3 in Arabic — Cnn From Scratch

Book slug : keras-3-arabic-professional
Chapter   : ch-20-ch20-cnn-from-scratch (lesson 20)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-20-ch20-cnn-from-scratch.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكة CNN من الصفر
Keras 3 in Arabic — Cnn From Scratch

Book slug : keras-3-arabic-professional
Chapter   : ch-20-ch20-cnn-from-scratch (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-20-ch20-cnn-from-scratch.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكة CNN من الصفر
Keras 3 in Arabic — Cnn From Scratch

Book slug : keras-3-arabic-professional
Chapter   : ch-20-ch20-cnn-from-scratch (lesson 20)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch20_cnn_from_scratch/001_s20_5_keras_pipeline_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.5 التنفيذ في Keras: خطّ كامل من الصفر
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch20_cnn_from_scratch/002_s20_6_tta_augmentation_test_time_test.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.6 تعزيز زمن الاختبار (TTA)
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:137
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch20_cnn_from_scratch/003_tta_test.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.5 التنفيذ في Keras: خطّ كامل من الصفر
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.6 تعزيز زمن الاختبار (TTA)
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:137
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكة CNN من الصفر
Keras 3 in Arabic — Cnn From Scratch

Book slug : keras-3-arabic-professional
Chapter   : ch-20-ch20-cnn-from-scratch (lesson 20)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch20_cnn_from_scratch/001_s20_5_keras_pipeline_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.5 التنفيذ في Keras: خطّ كامل من الصفر
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch20_cnn_from_scratch/002_s20_6_tta_augmentation_test_time_test.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.6 تعزيز زمن الاختبار (TTA)
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:137
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch20_cnn_from_scratch/003_tta_test.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch20_cnn_from_scratch.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.5 التنفيذ في Keras: خطّ كامل من الصفر
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.6 تعزيز زمن الاختبار (TTA)
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:137
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-20-ch20-cnn-from-scratch.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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
