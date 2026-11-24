#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
Keras 3 in Arabic — Optimizers

Book slug : keras-3-arabic-professional
Chapter   : ch-09-ch09-optimizers (lesson 9)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-09-ch09-optimizers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
Keras 3 in Arabic — Optimizers

Book slug : keras-3-arabic-professional
Chapter   : ch-09-ch09-optimizers (lesson 9)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-09-ch09-optimizers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
Keras 3 in Arabic — Optimizers

Book slug : keras-3-arabic-professional
Chapter   : ch-09-ch09-optimizers (lesson 9)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch09_optimizers/001_s9_6_adamw.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.6 AdamW: فصل اضمحلال الوزن
# المصدر  : src/chapters/ch09_optimizers.md:107
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch09_optimizers/002_keras_schedules.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : الجداول في Keras 3
# المصدر  : src/chapters/ch09_optimizers.md:132
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch09_optimizers/003_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : كيف نمرّرها؟
# المصدر  : src/chapters/ch09_optimizers.md:163
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch09_optimizers/004_trap_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : ⚠ فخّ مهم: الخطوات لا الحقبات
# المصدر  : src/chapters/ch09_optimizers.md:172
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch09_optimizers/005_s9_10_example_solved_diagnosis_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.10 مثال محلول: تشخيص تدرّب لا يتقارب
# المصدر  : src/chapters/ch09_optimizers.md:210
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.6 AdamW: فصل اضمحلال الوزن
# المصدر  : src/chapters/ch09_optimizers.md:107
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : الجداول في Keras 3
# المصدر  : src/chapters/ch09_optimizers.md:132
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : كيف نمرّرها؟
# المصدر  : src/chapters/ch09_optimizers.md:163
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : ⚠ فخّ مهم: الخطوات لا الحقبات
# المصدر  : src/chapters/ch09_optimizers.md:172
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.10 مثال محلول: تشخيص تدرّب لا يتقارب
# المصدر  : src/chapters/ch09_optimizers.md:210
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
Keras 3 in Arabic — Optimizers

Book slug : keras-3-arabic-professional
Chapter   : ch-09-ch09-optimizers (lesson 9)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch09_optimizers/001_s9_6_adamw.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.6 AdamW: فصل اضمحلال الوزن
# المصدر  : src/chapters/ch09_optimizers.md:107
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch09_optimizers/002_keras_schedules.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : الجداول في Keras 3
# المصدر  : src/chapters/ch09_optimizers.md:132
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch09_optimizers/003_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : كيف نمرّرها؟
# المصدر  : src/chapters/ch09_optimizers.md:163
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch09_optimizers/004_trap_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : ⚠ فخّ مهم: الخطوات لا الحقبات
# المصدر  : src/chapters/ch09_optimizers.md:172
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch09_optimizers/005_s9_10_example_solved_diagnosis_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.10 مثال محلول: تشخيص تدرّب لا يتقارب
# المصدر  : src/chapters/ch09_optimizers.md:210
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch09_optimizers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.6 AdamW: فصل اضمحلال الوزن
# المصدر  : src/chapters/ch09_optimizers.md:107
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : الجداول في Keras 3
# المصدر  : src/chapters/ch09_optimizers.md:132
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : كيف نمرّرها؟
# المصدر  : src/chapters/ch09_optimizers.md:163
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : ⚠ فخّ مهم: الخطوات لا الحقبات
# المصدر  : src/chapters/ch09_optimizers.md:172
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.10 مثال محلول: تشخيص تدرّب لا يتقارب
# المصدر  : src/chapters/ch09_optimizers.md:210
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# 48,000 عيّنة، batch_size=64، 10 حقبات
steps_per_epoch = 48000 // 64      # = 750
total_steps = 750 * 10             # = 7500

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-09-ch09-optimizers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")
