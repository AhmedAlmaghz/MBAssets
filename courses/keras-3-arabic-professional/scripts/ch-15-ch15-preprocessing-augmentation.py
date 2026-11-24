#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
Keras 3 in Arabic — Preprocessing Augmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-15-ch15-preprocessing-augmentation (lesson 15)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-15-ch15-preprocessing-augmentation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
Keras 3 in Arabic — Preprocessing Augmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-15-ch15-preprocessing-augmentation (lesson 15)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-15-ch15-preprocessing-augmentation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
Keras 3 in Arabic — Preprocessing Augmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-15-ch15-preprocessing-augmentation (lesson 15)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch15_preprocessing_augmentation/001_pipeline_processing_preprocessing_end_to_end.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : خط معالجة مسبق متكامل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:74
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch15_preprocessing_augmentation/002_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:201
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch15_preprocessing_augmentation/003_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:216
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch15_preprocessing_augmentation/004_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:224
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch15_preprocessing_augmentation/005_s15_6_errors_common.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.6 أخطاء شائعة في هذا الفصل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:249
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : خط معالجة مسبق متكامل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:74
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:201
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:216
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:224
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.6 أخطاء شائعة في هذا الفصل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:249
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
Keras 3 in Arabic — Preprocessing Augmentation

Book slug : keras-3-arabic-professional
Chapter   : ch-15-ch15-preprocessing-augmentation (lesson 15)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch15_preprocessing_augmentation/001_pipeline_processing_preprocessing_end_to_end.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : خط معالجة مسبق متكامل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:74
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch15_preprocessing_augmentation/002_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:201
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch15_preprocessing_augmentation/003_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:216
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch15_preprocessing_augmentation/004_s15_5_example_solved_skin.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:224
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch15_preprocessing_augmentation/005_s15_6_errors_common.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.6 أخطاء شائعة في هذا الفصل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:249
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch15_preprocessing_augmentation.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : خط معالجة مسبق متكامل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:74
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:201
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:216
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:224
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.6 أخطاء شائعة في هذا الفصل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:249
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
keras.layers.MixUp(alpha=0.3, name="mixup")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np

counts = np.array([5200, 1800, 700, 300])
weights = counts.sum() / (len(counts) * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print(class_weight)   # النتيجة المتوقعة: {0: 0.577, 1: 1.667, 2: 4.286, 3: 10.0}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-15-ch15-preprocessing-augmentation.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم
