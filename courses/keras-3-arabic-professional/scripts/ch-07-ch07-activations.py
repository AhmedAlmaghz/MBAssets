#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال التنشيط: أين يموت التدرّج؟
Keras 3 in Arabic — Activations

Book slug : keras-3-arabic-professional
Chapter   : ch-07-ch07-activations (lesson 7)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-07-ch07-activations.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال التنشيط: أين يموت التدرّج؟
Keras 3 in Arabic — Activations

Book slug : keras-3-arabic-professional
Chapter   : ch-07-ch07-activations (lesson 7)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-07-ch07-activations.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال التنشيط: أين يموت التدرّج؟
Keras 3 in Arabic — Activations

Book slug : keras-3-arabic-professional
Chapter   : ch-07-ch07-activations (lesson 7)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch07_activations/001_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:142
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch07_activations/002_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:161
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch07_activations/003_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:178
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch07_activations/004_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:207
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch07_activations/005_softmax_logits_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : حساب softmax يدويًّا وربطه بـ logits
# المصدر  : src/chapters/ch07_activations.md:247
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch07_activations/006_s7_5_example_solved_detection_fraud.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:300
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch07_activations/007_s7_5_example_solved_detection_fraud.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:342
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch07_activations/008_s7_6_errors_common.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.6 أخطاء شائعة
# المصدر  : src/chapters/ch07_activations.md:356
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch07_activations/009_s7_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.7 تمارين
# المصدر  : src/chapters/ch07_activations.md:391
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:142
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:161
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:178
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:207
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : حساب softmax يدويًّا وربطه بـ logits
# المصدر  : src/chapters/ch07_activations.md:247
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:300
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:342
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.6 أخطاء شائعة
# المصدر  : src/chapters/ch07_activations.md:356
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.7 تمارين
# المصدر  : src/chapters/ch07_activations.md:391
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال التنشيط: أين يموت التدرّج؟
Keras 3 in Arabic — Activations

Book slug : keras-3-arabic-professional
Chapter   : ch-07-ch07-activations (lesson 7)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch07_activations/001_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:142
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch07_activations/002_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:161
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch07_activations/003_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:178
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch07_activations/004_s7_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:207
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch07_activations/005_softmax_logits_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : حساب softmax يدويًّا وربطه بـ logits
# المصدر  : src/chapters/ch07_activations.md:247
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch07_activations/006_s7_5_example_solved_detection_fraud.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:300
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch07_activations/007_s7_5_example_solved_detection_fraud.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:342
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch07_activations/008_s7_6_errors_common.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.6 أخطاء شائعة
# المصدر  : src/chapters/ch07_activations.md:356
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch07_activations/009_s7_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.7 تمارين
# المصدر  : src/chapters/ch07_activations.md:391
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch07_activations.md
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:142
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:161
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:178
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:207
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : حساب softmax يدويًّا وربطه بـ logits
# المصدر  : src/chapters/ch07_activations.md:247
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:300
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:342
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.6 أخطاء شائعة
# المصدر  : src/chapters/ch07_activations.md:356
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.7 تمارين
# المصدر  : src/chapters/ch07_activations.md:391
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# قرار معاياري: argmax على الاحتمالات
decisions = ops.argmax(probs, axis=-1)
print(decisions.numpy())   # النتيجة المتوقعة: أرقام الفئات المتوقعة

# سجلّ درجة الثقة مقابل احتمال الاحتيال (الفئة 2)
confidence = probs[:, 2]
print("متوسط الثقة:", float(ops.mean(confidence)))

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-07-ch07-activations.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")
