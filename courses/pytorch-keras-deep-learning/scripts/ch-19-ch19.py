#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Keras 3: الفلسفة والمعمارية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 19 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-19-ch19 (lesson 19)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-19-ch19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Keras 3: الفلسفة والمعمارية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 19 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-19-ch19 (lesson 19)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch19.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # يجب أن يسبق استيراد keras

import keras

print(keras.backend.backend())      # الناتج: torch
print(keras.__version__)             # الناتج: 3.15.1

# keras.ops: واجهة شبيهة بـ numpy، تُنفَّذ على نواة Keras المختارة
x = keras.ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
y = keras.ops.matmul(x, keras.ops.transpose(x))
print(y.shape)                       # الناتج: (2, 2)
print(keras.ops.numpy(y))            # الناتج: [[17. 22.] [22. 29.]]

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch19.html
# ────────────────────────────────────────────────────────────────────────
import keras

keras.utils.set_random_seed(42)

# دالّة بلا حالة: نفس البذرة تعطي نفس القيم دائماً
a = keras.random.normal(shape=(2, 3), seed=42)
b = keras.random.normal(shape=(2, 3), seed=42)
print(keras.ops.all(keras.ops.equal(a, b)))     # الناتج: True

# مولّد ببذرة: يتقدّم، فتعطي قيماً مختلفة في كل استدعاء
gen = keras.random.SeedGenerator(seed=123)
c = keras.random.normal(shape=(2, 3), seed=gen)
d = keras.random.normal(shape=(2, 3), seed=gen)
print(keras.ops.all(keras.ops.equal(c, d)))     # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch19.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الأول: تتابع — سلسلة خطّية
seq = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),
])

# النمط الثاني: وظيفي — رسم بياني بفرعين يلتقيان
inp = layers.Input(shape=(784,))
h = layers.Dense(128, activation="relu")(inp)
branch_a = layers.Dense(32, activation="relu")(h)
branch_b = layers.Dense(32, activation="relu")(h)
merged = layers.Concatenate()([branch_a, branch_b])
func = keras.Model(inputs=inp, outputs=merged)

# النمط الثالث: وراثة — منطق داخل call
class SmallNet(keras.Model):
    def __init__(self, units=128):
        super().__init__()
        self.hidden = layers.Dense(units, activation="relu")
        self.out = layers.Dense(10)

    def call(self, inputs):
        return self.out(self.hidden(inputs))

sub = SmallNet()
sub.build((None, 784))               # بناء صريح ليعرف Keras الأبعاد

for name, m in (("sequential", seq), ("functional", func), ("subclass", sub)):
    print(name, m.count_params())    # الناتج: sequential 101770 functional 115818 subclass 101770

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch19.html
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam", loss="mse")
model.fit([[1., 2., 3., 4.], [0., 1., 2., 3.]],
          [[0., 1.], [1., 0.]], epochs=1, verbose=0)

# صيغة Keras 3: ملف .keras واحد يضمّ الوزنات والإعدادات والبيانات الوصفية
keras.saving.save_model(model, "toy.keras")
loaded = keras.saving.load_model("toy.keras")
print(loaded.count_params())          # الناتج: 10
print(loaded.optimizer is not None)   # الناتج: True

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch19.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
import torch

dense = keras.layers.Dense(4, activation="relu")
norm = keras.layers.LayerNormalization()

x_np = np.random.rand(2, 8).astype("float32")
out_np = dense(x_np)
print(type(out_np))                       # الناتج: <class 'torch.Tensor'>

x_t = torch.rand(2, 8)
out_t = dense(x_t)
print(type(out_t))                        # الناتج: <class 'torch.Tensor'>

# Keras تحوّل مُدخلات بايثون تلقائياً إلى كائنات النواة
y = norm(dense(x_np))
print(y.shape)                             # الناتج: (2, 4)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-19-ch19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # يجب أن يسبق استيراد keras

import keras

print(keras.backend.backend())      # الناتج: torch
print(keras.__version__)             # الناتج: 3.15.1

# keras.ops: واجهة شبيهة بـ numpy، تُنفَّذ على نواة Keras المختارة
x = keras.ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
y = keras.ops.matmul(x, keras.ops.transpose(x))
print(y.shape)                       # الناتج: (2, 2)
print(keras.ops.numpy(y))            # الناتج: [[17. 22.] [22. 29.]]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-19-ch19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras

keras.utils.set_random_seed(42)

# دالّة بلا حالة: نفس البذرة تعطي نفس القيم دائماً
a = keras.random.normal(shape=(2, 3), seed=42)
b = keras.random.normal(shape=(2, 3), seed=42)
print(keras.ops.all(keras.ops.equal(a, b)))     # الناتج: True

# مولّد ببذرة: يتقدّم، فتعطي قيماً مختلفة في كل استدعاء
gen = keras.random.SeedGenerator(seed=123)
c = keras.random.normal(shape=(2, 3), seed=gen)
d = keras.random.normal(shape=(2, 3), seed=gen)
print(keras.ops.all(keras.ops.equal(c, d)))     # الناتج: False

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-19-ch19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الأول: تتابع — سلسلة خطّية
seq = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),
])

# النمط الثاني: وظيفي — رسم بياني بفرعين يلتقيان
inp = layers.Input(shape=(784,))
h = layers.Dense(128, activation="relu")(inp)
branch_a = layers.Dense(32, activation="relu")(h)
branch_b = layers.Dense(32, activation="relu")(h)
merged = layers.Concatenate()([branch_a, branch_b])
func = keras.Model(inputs=inp, outputs=merged)

# النمط الثالث: وراثة — منطق داخل call
class SmallNet(keras.Model):
    def __init__(self, units=128):
        super().__init__()
        self.hidden = layers.Dense(units, activation="relu")
        self.out = layers.Dense(10)

    def call(self, inputs):
        return self.out(self.hidden(inputs))

sub = SmallNet()
sub.build((None, 784))               # بناء صريح ليعرف Keras الأبعاد

for name, m in (("sequential", seq), ("functional", func), ("subclass", sub)):
    print(name, m.count_params())    # الناتج: sequential 101770 functional 115818 subclass 101770

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-19-ch19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam", loss="mse")
model.fit([[1., 2., 3., 4.], [0., 1., 2., 3.]],
          [[0., 1.], [1., 0.]], epochs=1, verbose=0)

# صيغة Keras 3: ملف .keras واحد يضمّ الوزنات والإعدادات والبيانات الوصفية
keras.saving.save_model(model, "toy.keras")
loaded = keras.saving.load_model("toy.keras")
print(loaded.count_params())          # الناتج: 10
print(loaded.optimizer is not None)   # الناتج: True

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-19-ch19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
import torch

dense = keras.layers.Dense(4, activation="relu")
norm = keras.layers.LayerNormalization()

x_np = np.random.rand(2, 8).astype("float32")
out_np = dense(x_np)
print(type(out_np))                       # الناتج: <class 'torch.Tensor'>

x_t = torch.rand(2, 8)
out_t = dense(x_t)
print(type(out_t))                        # الناتج: <class 'torch.Tensor'>

# Keras تحوّل مُدخلات بايثون تلقائياً إلى كائنات النواة
y = norm(dense(x_np))
print(y.shape)                             # الناتج: (2, 4)
