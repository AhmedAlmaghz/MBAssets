#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تسريع التدريب
Keras 3 in Arabic — Acceleration

Book slug : keras-3-arabic-professional
Chapter   : ch-36-ch36-acceleration (lesson 36)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-36-ch36-acceleration.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تسريع التدريب
Keras 3 in Arabic — Acceleration

Book slug : keras-3-arabic-professional
Chapter   : ch-36-ch36-acceleration (lesson 36)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-36-ch36-acceleration.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تسريع التدريب
Keras 3 in Arabic — Acceleration

Book slug : keras-3-arabic-professional
Chapter   : ch-36-ch36-acceleration (lesson 36)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch36_acceleration/001_s36_2_precision_mixed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.2 تفعيل الدقّة المختلطة
# المصدر  : src/chapters/ch36_acceleration.md:28
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch36_acceleration/002_s36_3_xla_jit.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.3 XLA / JIT
# المصدر  : src/chapters/ch36_acceleration.md:62
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch36_acceleration/003_s36_4_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.4 تحسين خطّ البيانات
# المصدر  : src/chapters/ch36_acceleration.md:85
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch36_acceleration/004_s36_5_gradient_accumulation_compile.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.5 التجميع بالتدرّج (Gradient Accumulation)
# المصدر  : src/chapters/ch36_acceleration.md:124
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch36_acceleration/005_s36_8_multi_device_training_distributed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.8 التدريب الموزّع (Multi-Device)
# المصدر  : src/chapters/ch36_acceleration.md:167
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch36_acceleration/006_s36_9_profiling_measure_performance.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.9 قياس الأداء (Profiling)
# المصدر  : src/chapters/ch36_acceleration.md:200
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.2 تفعيل الدقّة المختلطة
# المصدر  : src/chapters/ch36_acceleration.md:28
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.3 XLA / JIT
# المصدر  : src/chapters/ch36_acceleration.md:62
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.4 تحسين خطّ البيانات
# المصدر  : src/chapters/ch36_acceleration.md:85
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.5 التجميع بالتدرّج (Gradient Accumulation)
# المصدر  : src/chapters/ch36_acceleration.md:124
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.8 التدريب الموزّع (Multi-Device)
# المصدر  : src/chapters/ch36_acceleration.md:167
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.9 قياس الأداء (Profiling)
# المصدر  : src/chapters/ch36_acceleration.md:200
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تسريع التدريب
Keras 3 in Arabic — Acceleration

Book slug : keras-3-arabic-professional
Chapter   : ch-36-ch36-acceleration (lesson 36)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch36_acceleration/001_s36_2_precision_mixed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.2 تفعيل الدقّة المختلطة
# المصدر  : src/chapters/ch36_acceleration.md:28
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch36_acceleration/002_s36_3_xla_jit.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.3 XLA / JIT
# المصدر  : src/chapters/ch36_acceleration.md:62
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch36_acceleration/003_s36_4_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.4 تحسين خطّ البيانات
# المصدر  : src/chapters/ch36_acceleration.md:85
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch36_acceleration/004_s36_5_gradient_accumulation_compile.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.5 التجميع بالتدرّج (Gradient Accumulation)
# المصدر  : src/chapters/ch36_acceleration.md:124
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch36_acceleration/005_s36_8_multi_device_training_distributed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.8 التدريب الموزّع (Multi-Device)
# المصدر  : src/chapters/ch36_acceleration.md:167
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch36_acceleration/006_s36_9_profiling_measure_performance.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.9 قياس الأداء (Profiling)
# المصدر  : src/chapters/ch36_acceleration.md:200
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch36_acceleration.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.2 تفعيل الدقّة المختلطة
# المصدر  : src/chapters/ch36_acceleration.md:28
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.3 XLA / JIT
# المصدر  : src/chapters/ch36_acceleration.md:62
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.4 تحسين خطّ البيانات
# المصدر  : src/chapters/ch36_acceleration.md:85
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.5 التجميع بالتدرّج (Gradient Accumulation)
# المصدر  : src/chapters/ch36_acceleration.md:124
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.8 التدريب الموزّع (Multi-Device)
# المصدر  : src/chapters/ch36_acceleration.md:167
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.9 قياس الأداء (Profiling)
# المصدر  : src/chapters/ch36_acceleration.md:200
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    jit_compile=True,       # تفعيل XLA
)
# النتيجة المتوقعة: تسريع 10–30% على النماذج ذات الرسوم البيانيّة الكبيرة.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-36-ch36-acceleration.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.
