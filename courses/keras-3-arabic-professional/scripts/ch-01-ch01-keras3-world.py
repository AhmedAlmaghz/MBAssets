#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — Keras 3: كيف تغيّر عالَم التعلّم العميق؟
Keras 3 in Arabic — Keras3 World

Book slug : keras-3-arabic-professional
Chapter   : ch-01-ch01-keras3-world (lesson 1)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-01-ch01-keras3-world.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — Keras 3: كيف تغيّر عالَم التعلّم العميق؟
Keras 3 in Arabic — Keras3 World

Book slug : keras-3-arabic-professional
Chapter   : ch-01-ch01-keras3-world (lesson 1)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-01-ch01-keras3-world.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — Keras 3: كيف تغيّر عالَم التعلّم العميق؟
Keras 3 in Arabic — Keras3 World

Book slug : keras-3-arabic-professional
Chapter   : ch-01-ch01-keras3-world (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch01_keras3_world/001_s1_2_structure_multi_backends_backend.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.2 البنية متعددة الواجهات الخلفية
# المصدر  : src/chapters/ch01_keras3_world.md:35
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch01_keras3_world/002_s1_7_example_solved_upgrade_system.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.7 مثال محلول: ترقية نظام كشف الاحتيال
# المصدر  : src/chapters/ch01_keras3_world.md:139
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch01_keras3_world.md
# ────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch01_keras3_world.md
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.2 البنية متعددة الواجهات الخلفية
# المصدر  : src/chapters/ch01_keras3_world.md:35
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.7 مثال محلول: ترقية نظام كشف الاحتيال
# المصدر  : src/chapters/ch01_keras3_world.md:139
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — Keras 3: كيف تغيّر عالَم التعلّم العميق؟
Keras 3 in Arabic — Keras3 World

Book slug : keras-3-arabic-professional
Chapter   : ch-01-ch01-keras3-world (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch01_keras3_world/001_s1_2_structure_multi_backends_backend.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.2 البنية متعددة الواجهات الخلفية
# المصدر  : src/chapters/ch01_keras3_world.md:35
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch01_keras3_world/002_s1_7_example_solved_upgrade_system.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.7 مثال محلول: ترقية نظام كشف الاحتيال
# المصدر  : src/chapters/ch01_keras3_world.md:139
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch01_keras3_world.md
# ────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch01_keras3_world.md
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.2 البنية متعددة الواجهات الخلفية
# المصدر  : src/chapters/ch01_keras3_world.md:35
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.7 مثال محلول: ترقية نظام كشف الاحتيال
# المصدر  : src/chapters/ch01_keras3_world.md:139
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "jax"  # اختيار المحرّك قبل استيراد keras

import keras  # الاستيراد العام: لا tensorflow ولا torch

print("إصدار Keras:", keras.__version__)      # النتيجة المتوقعة: 3.15.1
print("الواجهة الخلفية:", keras.backend.backend())  # النتيجة المتوقعة: jax

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-01-ch01-keras3-world.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense
