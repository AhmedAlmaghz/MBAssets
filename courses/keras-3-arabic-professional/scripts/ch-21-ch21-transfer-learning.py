#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نقل المعرفة والتدريب الدقيق
Keras 3 in Arabic — Transfer Learning

Book slug : keras-3-arabic-professional
Chapter   : ch-21-ch21-transfer-learning (lesson 21)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-21-ch21-transfer-learning.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نقل المعرفة والتدريب الدقيق
Keras 3 in Arabic — Transfer Learning

Book slug : keras-3-arabic-professional
Chapter   : ch-21-ch21-transfer-learning (lesson 21)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-21-ch21-transfer-learning.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نقل المعرفة والتدريب الدقيق
Keras 3 in Arabic — Transfer Learning

Book slug : keras-3-arabic-professional
Chapter   : ch-21-ch21-transfer-learning (lesson 21)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch21_transfer_learning/001_s21_3_rate_learning_discriminative.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.3 معدّل التعلّم التمييزي
# المصدر  : src/chapters/ch21_transfer_learning.md:39
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch21_transfer_learning/002_s21_5_head_classification.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.5 رأس التصنيف: ما الذي تستبدله؟
# المصدر  : src/chapters/ch21_transfer_learning.md:90
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch21_transfer_learning/003_measure_transactions_before_unfreezing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : قياس المعاملات القابلة للتدريب قبل وبعد الفكّ
# المصدر  : src/chapters/ch21_transfer_learning.md:131
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.3 معدّل التعلّم التمييزي
# المصدر  : src/chapters/ch21_transfer_learning.md:39
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.5 رأس التصنيف: ما الذي تستبدله؟
# المصدر  : src/chapters/ch21_transfer_learning.md:90
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : قياس المعاملات القابلة للتدريب قبل وبعد الفكّ
# المصدر  : src/chapters/ch21_transfer_learning.md:131
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نقل المعرفة والتدريب الدقيق
Keras 3 in Arabic — Transfer Learning

Book slug : keras-3-arabic-professional
Chapter   : ch-21-ch21-transfer-learning (lesson 21)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch21_transfer_learning/001_s21_3_rate_learning_discriminative.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.3 معدّل التعلّم التمييزي
# المصدر  : src/chapters/ch21_transfer_learning.md:39
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch21_transfer_learning/002_s21_5_head_classification.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.5 رأس التصنيف: ما الذي تستبدله؟
# المصدر  : src/chapters/ch21_transfer_learning.md:90
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch21_transfer_learning/003_measure_transactions_before_unfreezing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : قياس المعاملات القابلة للتدريب قبل وبعد الفكّ
# المصدر  : src/chapters/ch21_transfer_learning.md:131
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch21_transfer_learning.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.3 معدّل التعلّم التمييزي
# المصدر  : src/chapters/ch21_transfer_learning.md:39
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.5 رأس التصنيف: ما الذي تستبدله؟
# المصدر  : src/chapters/ch21_transfer_learning.md:90
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : قياس المعاملات القابلة للتدريب قبل وبعد الفكّ
# المصدر  : src/chapters/ch21_transfer_learning.md:131
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-21-ch21-transfer-learning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.
