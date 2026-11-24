#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوتور: اللغة التي يفكّر بها Keras
Keras 3 in Arabic — Tensors

Book slug : keras-3-arabic-professional
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 24 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-03-ch03-tensors.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوتور: اللغة التي يفكّر بها Keras
Keras 3 in Arabic — Tensors

Book slug : keras-3-arabic-professional
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 23 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-03-ch03-tensors.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوتور: اللغة التي يفكّر بها Keras
Keras 3 in Arabic — Tensors

Book slug : keras-3-arabic-professional
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch03_tensors/002_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : النوع والجهاز
# المصدر  : src/chapters/ch03_tensors.md:48
# الترتيب : 2/12
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch03_tensors/003_example.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : مثال عملي: إضافة متجه إلى مصفوفة
# المصدر  : src/chapters/ch03_tensors.md:81
# الترتيب : 3/12
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch03_tensors/004_state.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : الحالة التي تفاجئ الجميع
# المصدر  : src/chapters/ch03_tensors.md:99
# الترتيب : 4/12
# ──────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch03_tensors/005_broadcasting.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : متى يفشل البث؟
# المصدر  : src/chapters/ch03_tensors.md:113
# الترتيب : 5/12
# ──────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch03_tensors/006_rule.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القاعدة العملية
# المصدر  : src/chapters/ch03_tensors.md:125
# الترتيب : 6/12
# ──────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch03_tensors/007_graph.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:171
# الترتيب : 7/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch03_tensors/008_graph.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:188
# الترتيب : 8/12
# ──────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch03_tensors/009_s3_4_keras_ops.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.4 `keras.ops` ولِماذا وُجدت؟
# المصدر  : src/chapters/ch03_tensors.md:216
# الترتيب : 9/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch03_tensors/010_s3_5_variables_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.5 المتغيّرات: ما الذي يتغيّر أثناء التدريب؟
# المصدر  : src/chapters/ch03_tensors.md:236
# الترتيب : 10/12
# ──────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch03_tensors/011_learnable.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القابل للتعلّم وغير القابل
# المصدر  : src/chapters/ch03_tensors.md:246
# الترتيب : 11/12
# ──────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch03_tensors/012_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف يُحدَّث المتغيّر؟
# المصدر  : src/chapters/ch03_tensors.md:266
# الترتيب : 12/12
# ──────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [021] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [022] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : النوع والجهاز
# المصدر  : src/chapters/ch03_tensors.md:48
# الترتيب : 2/12
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : مثال عملي: إضافة متجه إلى مصفوفة
# المصدر  : src/chapters/ch03_tensors.md:81
# الترتيب : 3/12
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : الحالة التي تفاجئ الجميع
# المصدر  : src/chapters/ch03_tensors.md:99
# الترتيب : 4/12
# ──────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : متى يفشل البث؟
# المصدر  : src/chapters/ch03_tensors.md:113
# الترتيب : 5/12
# ──────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القاعدة العملية
# المصدر  : src/chapters/ch03_tensors.md:125
# الترتيب : 6/12
# ──────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:171
# الترتيب : 7/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:188
# الترتيب : 8/12
# ──────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.4 `keras.ops` ولِماذا وُجدت؟
# المصدر  : src/chapters/ch03_tensors.md:216
# الترتيب : 9/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.5 المتغيّرات: ما الذي يتغيّر أثناء التدريب؟
# المصدر  : src/chapters/ch03_tensors.md:236
# الترتيب : 10/12
# ──────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القابل للتعلّم وغير القابل
# المصدر  : src/chapters/ch03_tensors.md:246
# الترتيب : 11/12
# ──────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف يُحدَّث المتغيّر؟
# المصدر  : src/chapters/ch03_tensors.md:266
# الترتيب : 12/12
# ──────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوتور: اللغة التي يفكّر بها Keras
Keras 3 in Arabic — Tensors

Book slug : keras-3-arabic-professional
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch03_tensors/002_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : النوع والجهاز
# المصدر  : src/chapters/ch03_tensors.md:48
# الترتيب : 2/12
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch03_tensors/003_example.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : مثال عملي: إضافة متجه إلى مصفوفة
# المصدر  : src/chapters/ch03_tensors.md:81
# الترتيب : 3/12
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch03_tensors/004_state.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : الحالة التي تفاجئ الجميع
# المصدر  : src/chapters/ch03_tensors.md:99
# الترتيب : 4/12
# ──────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch03_tensors/005_broadcasting.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : متى يفشل البث؟
# المصدر  : src/chapters/ch03_tensors.md:113
# الترتيب : 5/12
# ──────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch03_tensors/006_rule.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القاعدة العملية
# المصدر  : src/chapters/ch03_tensors.md:125
# الترتيب : 6/12
# ──────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch03_tensors/007_graph.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:171
# الترتيب : 7/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch03_tensors/008_graph.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:188
# الترتيب : 8/12
# ──────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch03_tensors/009_s3_4_keras_ops.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.4 `keras.ops` ولِماذا وُجدت؟
# المصدر  : src/chapters/ch03_tensors.md:216
# الترتيب : 9/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch03_tensors/010_s3_5_variables_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.5 المتغيّرات: ما الذي يتغيّر أثناء التدريب؟
# المصدر  : src/chapters/ch03_tensors.md:236
# الترتيب : 10/12
# ──────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch03_tensors/011_learnable.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القابل للتعلّم وغير القابل
# المصدر  : src/chapters/ch03_tensors.md:246
# الترتيب : 11/12
# ──────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch03_tensors/012_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف يُحدَّث المتغيّر؟
# المصدر  : src/chapters/ch03_tensors.md:266
# الترتيب : 12/12
# ──────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [021] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [022] src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : النوع والجهاز
# المصدر  : src/chapters/ch03_tensors.md:48
# الترتيب : 2/12
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : مثال عملي: إضافة متجه إلى مصفوفة
# المصدر  : src/chapters/ch03_tensors.md:81
# الترتيب : 3/12
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : الحالة التي تفاجئ الجميع
# المصدر  : src/chapters/ch03_tensors.md:99
# الترتيب : 4/12
# ──────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : متى يفشل البث؟
# المصدر  : src/chapters/ch03_tensors.md:113
# الترتيب : 5/12
# ──────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القاعدة العملية
# المصدر  : src/chapters/ch03_tensors.md:125
# الترتيب : 6/12
# ──────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:171
# الترتيب : 7/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:188
# الترتيب : 8/12
# ──────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.4 `keras.ops` ولِماذا وُجدت؟
# المصدر  : src/chapters/ch03_tensors.md:216
# الترتيب : 9/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : 3.5 المتغيّرات: ما الذي يتغيّر أثناء التدريب؟
# المصدر  : src/chapters/ch03_tensors.md:236
# الترتيب : 10/12
# ──────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القابل للتعلّم وغير القابل
# المصدر  : src/chapters/ch03_tensors.md:246
# الترتيب : 11/12
# ──────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف يُحدَّث المتغيّر؟
# المصدر  : src/chapters/ch03_tensors.md:266
# الترتيب : 12/12
# ──────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مصفوفة 3×2 ومتجه طول 2
a = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
b = np.array([10., 20.])                       # (2,)

# يعمل — لكن ليس بالطريقة التي يتوقّعها المبتدئ
print(a + b)
# [[11. 22.]
#  [13. 24.]
#  [15. 26.]]

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
a = np.array([[1.], [2.], [3.]])   # (3, 1) عمود
b = np.array([[10., 20.]])         # (1, 2) صف

print(a + b)
# [[11. 21.]
#  [12. 22.]
#  [13. 23.]]

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
a = np.random.rand(3, 1)    # (3, 1)
b = np.random.rand(4, 5)    # (4, 5)
# a + b  →  ValueError

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
x = np.random.rand(32, 784).astype("float32")   # 32 عيّنة، 784 خاصية

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.array([1.0, 2.0, 3.0])

keras.ops.softmax(x)      # تطبيع احتمالي
keras.ops.relu(x)         # تنشيط
keras.ops.matmul(x, x.T)  # ضرب مصفوفي
keras.ops.mean(x)         # متوسط
keras.ops.argmax(x)       # فهرس أكبر قيمة

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import keras

v = keras.Variable([1.0, 2.0, 3.0], name="weights")
print(v)              # <Variable shape=(3,), dtype=float32, value=[1., 2., 3.]>
print(v.shape, v.dtype)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/keras-3-arabic-professional/notebooks/ch-03-ch03-tensors.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
v.assign([4.0, 5.0, 6.0])     # استبدال كامل
v.assign_add([1.0, 1.0, 1.0]) # جمع
