#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الاشتقاق التلقائي والرسوم البيانية
Professional TensorFlow — Autodiff

Book slug : tensorflow-professional-ar
Chapter   : ch-04-ch04-autodiff (lesson 4)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-04-ch04-autodiff.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الاشتقاق التلقائي والرسوم البيانية
Professional TensorFlow — Autodiff

Book slug : tensorflow-professional-ar
Chapter   : ch-04-ch04-autodiff (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

weight = tf.Variable(3.0)
distance = tf.constant(2.0)
target = tf.constant(10.0)

with tf.GradientTape() as tape:
    prediction = weight * distance
    loss = tf.square(prediction - target)

gradient = tape.gradient(loss, weight)
print("prediction:", prediction.numpy())  # 6.0
print("loss:", loss.numpy())              # 16.0
print("d(loss)/d(weight):", gradient.numpy())  # -16.0

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# (1) متغيّر: الشريط يراقبه تلقائيًا
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3 + 2.0 * x            # y = 12 عند x = 2
print("dy/dx =", tape.gradient(y, x).numpy())   # 14.0  (3x² + 2)

# (2) ثابت: لا يُراقَب إلا بطلبٍ صريح
c = tf.constant(3.0)
with tf.GradientTape() as tape:
    y2 = c ** 2
print("بلا watch:", tape.gradient(y2, c))       # None

with tf.GradientTape() as tape:
    tape.watch(c)
    y2 = c ** 2
print("مع watch:", tape.gradient(y2, c).numpy())  # 6.0

# (3) شريط دائم: يسمح بأكثر من اشتقاق
with tf.GradientTape(persistent=True) as tape:
    a = tf.Variable(1.5)
    f = a ** 2
    g = a ** 3
print("df/da =", tape.gradient(f, a).numpy())   # 3.0
print("dg/da =", tape.gradient(g, a).numpy())   # 6.75
del tape                                        # حرّر الموارد

# (4) المشتقة الثانية: شريطان متداخلان
x2 = tf.Variable(2.0)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y3 = x2 ** 3
    first = inner.gradient(y3, x2)              # 12.0
second = outer.gradient(first, x2)
print("d²y/dx² =", second.numpy())              # 12.0

# (5) جاكوبي: دالة من R² إلى R³
v = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    out = tf.stack([v[0] ** 2, v[0] * v[1], v[1] ** 3])
J = tape.jacobian(out, v)
print("shape =", J.shape)                       # (3, 2)
print(J.numpy())
# [[ 4.  0.]
#  [ 3.  2.]
#  [ 0. 27.]]

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# بيانات: تكلفة الشحن تزيد خطيًا مع الوزن، مع ضجيج بسيط
rng = np.random.default_rng(42)
x = rng.uniform(0.5, 5.0, size=(200, 1)).astype("float32")
y = 4.0 * x + 2.5 + rng.normal(0.0, 0.8, size=(200, 1)).astype("float32")

# وسطان يُتعلّمان: نبدأ من الصفر
w = tf.Variable(0.0, name="w")
b = tf.Variable(0.0, name="b")
optimizer = keras.optimizers.SGD(learning_rate=0.02)

for epoch in range(200):
    # كل خطوة تحتاج شريطًا جديدًا: الشريط يُستهلك بعد gradient
    with tf.GradientTape() as tape:
        y_pred = w * x + b
        loss = tf.reduce_mean(tf.square(y_pred - y))

    # التدرّج بالنسبة إلى الوسيطين معًا
    dw, db = tape.gradient(loss, [w, b])

    # المحسّن يطبّق: variable -= learning_rate * gradient
    optimizer.apply_gradients(zip([dw, db], [w, b]))

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss = {loss.numpy():.3f}")

print("w =", float(w.numpy()), "| b =", float(b.numpy()))
# المتوقع: loss ينخفض من نحو 200 إلى أقل من 1،
# وw يقترب من 4.0، وb يقترب من 2.5

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def h(x):
    return tf.sin(x ** 2) * tf.exp(x / 3.0)

x0 = tf.Variable(1.2)
with tf.GradientTape() as tape:
    value = h(x0)
dy = tape.gradient(value, x0)

# تحقق عددي: (f(x+ε) - f(x-ε)) / 2ε
eps = 1e-3
numeric = (h(tf.constant(1.2 + eps)) - h(tf.constant(1.2 - eps))) / (2.0 * eps)

print("autodiff =", round(float(dy), 6))     # 0.95999
print("numeric  =", round(float(numeric), 6))  # 0.95999 تقريبًا
# المتوقع: القيمتان متطابقتان إلى خمس منازل

# ────────────────────────────────────────────────────────────────────────
# [005] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

traces = {"count": 0}

@tf.function
def scale(x):
    traces["count"] += 1                 # يُنفَّذ وقت التتبّع فقط
    tf.print("trace:", traces["count"])  # tf.print يُنفَّذ وقت التشغيل
    return x * 2.0 + 1.0

print(scale(tf.constant([1.0, 2.0])).numpy())        # [3. 5.]
print(scale(tf.constant([1.0, 2.0, 3.0])).numpy())   # [3. 5. 7.]
print("عدد التتبعات =", traces["count"])             # 2

# بصمة ثابتة: شكل مجهول في البُعد الأول، فتتبع واحد يكفي
traces2 = {"count": 0}

@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def scale_fixed(x):
    traces2["count"] += 1
    return x * 2.0 + 1.0

scale_fixed(tf.constant([1.0]))
scale_fixed(tf.constant([1.0, 2.0, 3.0]))
print("عدد التتبعات =", traces2["count"])            # 1

# reduce_retracing: اسمح بتخفيف الأشكال بدل رسمٍ لكل حجم
@tf.function(reduce_retracing=True)
def normalize(x):
    return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-7)

print(normalize(tf.constant([1.0, 2.0, 3.0])).numpy())
# المتوقع: [-1.2247  0.      1.2247]

# ────────────────────────────────────────────────────────────────────────
# [006] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
import time
import tensorflow as tf

# دالة كثيفة: ضرب مصفوفتين كبيرتين
@tf.function
def matmul_graph(a, b):
    return tf.linalg.matmul(a, b)

@tf.function(jit_compile=True)
def matmul_xla(a, b):
    return tf.linalg.matmul(a, b)

a = tf.random.normal((1024, 1024))
b = tf.random.normal((1024, 1024))

# تسخين: أول استدعاء يبني الرسم ويُصرّف النواة
matmul_graph(a, b)
matmul_xla(a, b)

def bench(fn, repeats=20):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(a, b)
    return (time.perf_counter() - start) / repeats

print("graph:", round(bench(matmul_graph) * 1000, 2), "ms")
print("xla  :", round(bench(matmul_xla) * 1000, 2), "ms")
# المتوقع: رقمان متقاربان على CPU؛ الفرق يظهر أوضح
# على GPU أو مع سلاسل طويلة من العمليات الصغيرة

# ────────────────────────────────────────────────────────────────────────
# [007] book/src/chapters/ch04_autodiff.md
# ────────────────────────────────────────────────────────────────────────
for batch in dataset:
    @tf.function                 # خطأ: زخرفة جديدة في كل تكرار
    def train_step(batch):
        ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

weight = tf.Variable(3.0)
distance = tf.constant(2.0)
target = tf.constant(10.0)

with tf.GradientTape() as tape:
    prediction = weight * distance
    loss = tf.square(prediction - target)

gradient = tape.gradient(loss, weight)
print("prediction:", prediction.numpy())  # 6.0
print("loss:", loss.numpy())              # 16.0
print("d(loss)/d(weight):", gradient.numpy())  # -16.0

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# (1) متغيّر: الشريط يراقبه تلقائيًا
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3 + 2.0 * x            # y = 12 عند x = 2
print("dy/dx =", tape.gradient(y, x).numpy())   # 14.0  (3x² + 2)

# (2) ثابت: لا يُراقَب إلا بطلبٍ صريح
c = tf.constant(3.0)
with tf.GradientTape() as tape:
    y2 = c ** 2
print("بلا watch:", tape.gradient(y2, c))       # None

with tf.GradientTape() as tape:
    tape.watch(c)
    y2 = c ** 2
print("مع watch:", tape.gradient(y2, c).numpy())  # 6.0

# (3) شريط دائم: يسمح بأكثر من اشتقاق
with tf.GradientTape(persistent=True) as tape:
    a = tf.Variable(1.5)
    f = a ** 2
    g = a ** 3
print("df/da =", tape.gradient(f, a).numpy())   # 3.0
print("dg/da =", tape.gradient(g, a).numpy())   # 6.75
del tape                                        # حرّر الموارد

# (4) المشتقة الثانية: شريطان متداخلان
x2 = tf.Variable(2.0)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y3 = x2 ** 3
    first = inner.gradient(y3, x2)              # 12.0
second = outer.gradient(first, x2)
print("d²y/dx² =", second.numpy())              # 12.0

# (5) جاكوبي: دالة من R² إلى R³
v = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    out = tf.stack([v[0] ** 2, v[0] * v[1], v[1] ** 3])
J = tape.jacobian(out, v)
print("shape =", J.shape)                       # (3, 2)
print(J.numpy())
# [[ 4.  0.]
#  [ 3.  2.]
#  [ 0. 27.]]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# بيانات: تكلفة الشحن تزيد خطيًا مع الوزن، مع ضجيج بسيط
rng = np.random.default_rng(42)
x = rng.uniform(0.5, 5.0, size=(200, 1)).astype("float32")
y = 4.0 * x + 2.5 + rng.normal(0.0, 0.8, size=(200, 1)).astype("float32")

# وسطان يُتعلّمان: نبدأ من الصفر
w = tf.Variable(0.0, name="w")
b = tf.Variable(0.0, name="b")
optimizer = keras.optimizers.SGD(learning_rate=0.02)

for epoch in range(200):
    # كل خطوة تحتاج شريطًا جديدًا: الشريط يُستهلك بعد gradient
    with tf.GradientTape() as tape:
        y_pred = w * x + b
        loss = tf.reduce_mean(tf.square(y_pred - y))

    # التدرّج بالنسبة إلى الوسيطين معًا
    dw, db = tape.gradient(loss, [w, b])

    # المحسّن يطبّق: variable -= learning_rate * gradient
    optimizer.apply_gradients(zip([dw, db], [w, b]))

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss = {loss.numpy():.3f}")

print("w =", float(w.numpy()), "| b =", float(b.numpy()))
# المتوقع: loss ينخفض من نحو 200 إلى أقل من 1،
# وw يقترب من 4.0، وb يقترب من 2.5

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def h(x):
    return tf.sin(x ** 2) * tf.exp(x / 3.0)

x0 = tf.Variable(1.2)
with tf.GradientTape() as tape:
    value = h(x0)
dy = tape.gradient(value, x0)

# تحقق عددي: (f(x+ε) - f(x-ε)) / 2ε
eps = 1e-3
numeric = (h(tf.constant(1.2 + eps)) - h(tf.constant(1.2 - eps))) / (2.0 * eps)

print("autodiff =", round(float(dy), 6))     # 0.95999
print("numeric  =", round(float(numeric), 6))  # 0.95999 تقريبًا
# المتوقع: القيمتان متطابقتان إلى خمس منازل

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

traces = {"count": 0}

@tf.function
def scale(x):
    traces["count"] += 1                 # يُنفَّذ وقت التتبّع فقط
    tf.print("trace:", traces["count"])  # tf.print يُنفَّذ وقت التشغيل
    return x * 2.0 + 1.0

print(scale(tf.constant([1.0, 2.0])).numpy())        # [3. 5.]
print(scale(tf.constant([1.0, 2.0, 3.0])).numpy())   # [3. 5. 7.]
print("عدد التتبعات =", traces["count"])             # 2

# بصمة ثابتة: شكل مجهول في البُعد الأول، فتتبع واحد يكفي
traces2 = {"count": 0}

@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def scale_fixed(x):
    traces2["count"] += 1
    return x * 2.0 + 1.0

scale_fixed(tf.constant([1.0]))
scale_fixed(tf.constant([1.0, 2.0, 3.0]))
print("عدد التتبعات =", traces2["count"])            # 1

# reduce_retracing: اسمح بتخفيف الأشكال بدل رسمٍ لكل حجم
@tf.function(reduce_retracing=True)
def normalize(x):
    return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-7)

print(normalize(tf.constant([1.0, 2.0, 3.0])).numpy())
# المتوقع: [-1.2247  0.      1.2247]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
import tensorflow as tf

# دالة كثيفة: ضرب مصفوفتين كبيرتين
@tf.function
def matmul_graph(a, b):
    return tf.linalg.matmul(a, b)

@tf.function(jit_compile=True)
def matmul_xla(a, b):
    return tf.linalg.matmul(a, b)

a = tf.random.normal((1024, 1024))
b = tf.random.normal((1024, 1024))

# تسخين: أول استدعاء يبني الرسم ويُصرّف النواة
matmul_graph(a, b)
matmul_xla(a, b)

def bench(fn, repeats=20):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(a, b)
    return (time.perf_counter() - start) / repeats

print("graph:", round(bench(matmul_graph) * 1000, 2), "ms")
print("xla  :", round(bench(matmul_xla) * 1000, 2), "ms")
# المتوقع: رقمان متقاربان على CPU؛ الفرق يظهر أوضح
# على GPU أو مع سلاسل طويلة من العمليات الصغيرة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/tensorflow-professional-ar/notebooks/ch-04-ch04-autodiff.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
for batch in dataset:
    @tf.function                 # خطأ: زخرفة جديدة في كل تكرار
    def train_step(batch):
        ...
