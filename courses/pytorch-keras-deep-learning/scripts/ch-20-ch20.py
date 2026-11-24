#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الطبقات والنماذج في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 20 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-20-ch20 (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-20-ch20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الطبقات والنماذج في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 20 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-20-ch20 (lesson 20)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# نفككBehaviour الطبقة الجاهزة لنرى كيف تتبع Keras العقد
dense = layers.Dense(8, activation="relu", name="hidden")

print("before build, built =", dense.built)          # الناتج: False
print("config =", dense.get_config()["activation"])   # الناتج: relu

# أول استدعاء يبني الطبقة تلقائياً
out = dense(keras.ops.zeros((2, 5)))
print("after call, built =", dense.built)             # الناتج: True
print("kernel shape =", dense.kernel.shape)           # الناتج: (5, 8)
print("output shape =", out.shape)                    # الناتج: (2, 8)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الوظيفي: Input ينشئ "مصفوفة رمزية" لا تحمل بيانات
inp = layers.Input(shape=(28, 28, 1), name="image")
x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
out = layers.Dense(10)(x)

model = keras.Model(inputs=inp, outputs=out)

# الشكل المتوقَّع مقابل الشكل الحقيقي
print("symbolic:", model.output_shape)                 # الناتج: (None, 10)
print("input shape:", model.input_shape)                # الناتج: (None, 28, 28, 1)

real = keras.ops.zeros((4, 28, 28, 1))
print("actual:", model(real).shape)                     # الناتج: (4, 10)
print("params:", model.count_params())                  # الناتج: 12710

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class WeightedScale(layers.Layer):
    """يضرب كلّ قناة في وزنٍ متعلَّم، ثم يضيف إزاحة متعلَّمة."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        # الأوزان تُنشأ هنا لا في __init__: الشكل صار معروفاً الآن
        self.gamma = self.add_weight(
            shape=(self.channels,),
            initializer="ones",
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
            name="beta",
        )
        super().build(input_shape)

    def call(self, inputs):
        # عمليات محايدة: تعمل على أيّ نواة
        return inputs * self.gamma + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


layer = WeightedScale(4, name="wscale")
out = layer(keras.ops.ones((3, 10, 4)))
print("output shape:", out.shape)                 # الناتج: (3, 10, 4)
print("trainable weights:", len(layer.trainable_weights))   # الناتج: 2
print("config:", layer.get_config()["channels"])   # الناتج: 4

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops

RECEIVED = {}


class MaskedMean(layers.Layer):
    """متوسّط حسابي يتجاهل الخطوات المقنّعة."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True      # إعلان: هذه الطبقة تفهم القناع

    def call(self, inputs, mask=None):
        RECEIVED["mask"] = mask
        if mask is None:
            return ops.mean(inputs, axis=1)
        m = ops.expand_dims(ops.cast(mask, inputs.dtype), axis=-1)
        total = ops.sum(inputs * m, axis=1)
        count = ops.maximum(ops.sum(m, axis=1), 1.0)
        return total / count

    def compute_mask(self, inputs, mask=None):
        return None                        # لا قناع صادر: انتهى الاستخدام هنا

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1])


# ثلاث خطوات: الثانية أصفار، أي أنها خطوة فارغة
x = keras.ops.convert_to_tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])

model = keras.Sequential([
    layers.Input(shape=(3, 2)),
    layers.Masking(mask_value=0.0),
    MaskedMean(),
])
result = model(x)
print("mask received:", keras.ops.numpy(RECEIVED["mask"]).tolist())   # الناتج: [[True, False, True]]
print("masked mean:", keras.ops.numpy(result).tolist())               # الناتج: [[1.5, 3.0]]

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# 1) تسجيل الطبقة المخصّصة ليعرفها Keras عند التحميل
@keras.saving.register_keras_serializable(package="Industrial", name="WeightedScale")
class WeightedScale(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.channels,), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(
            shape=(self.channels,), initializer="zeros", trainable=True, name="beta")
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


model = keras.Sequential([
    layers.Input(shape=(6,)),
    WeightedScale(6),
    layers.Dense(2),
])
model.compile(optimizer="adam", loss="mse")
model.fit(keras.ops.zeros((8, 6)), keras.ops.zeros((8, 2)), epochs=1, verbose=0)

# 2) الحفظ الكامل: بنية + أوزان + حالة المحرّك
keras.saving.save_model(model, "sensor.keras")
reloaded = keras.saving.load_model("sensor.keras")
print("reloaded params:", reloaded.count_params())

# 3) حفظ الأوزان فقط: أخفّ، يُستعمل مع بنية معرّفة في الكود
model.save_weights("sensor.weights.h5")
model.load_weights("sensor.weights.h5")

# 4) التصدير: صيغة تنفيذية مستقلة
model.export("sensor_export", format="torch")
print("exported:", sorted(__import__("os").listdir("sensor_export")))

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch20.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class GluonDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        in_dim = input_shape[-1] // 2
        self.left = self.add_weight(shape=(in_dim, self.units),
                                    initializer="glorot_uniform", name="left")
        self.right = self.add_weight(shape=(in_dim, self.units),
                                     initializer="glorot_uniform", name="right")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        super().build(input_shape)

    def call(self, inputs):
        a, b = ops.split(inputs, 2, axis=-1)
        return a * ops.sigmoid(b) + self.b

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


m = keras.Sequential([layers.Input(shape=(8,)), GluonDense(4)])
m.compile(optimizer="adam", loss="mse")
x = keras.ops.random.normal(shape=(4, 8))
before = keras.ops.numpy(m(x))
keras.saving.save_model(m, "gluon.keras")
after = keras.ops.numpy(keras.saving.load_model("gluon.keras")(x))
print("max diff:", float(abs(before - after).max()))   # الناتج: 0.0

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# نفككBehaviour الطبقة الجاهزة لنرى كيف تتبع Keras العقد
dense = layers.Dense(8, activation="relu", name="hidden")

print("before build, built =", dense.built)          # الناتج: False
print("config =", dense.get_config()["activation"])   # الناتج: relu

# أول استدعاء يبني الطبقة تلقائياً
out = dense(keras.ops.zeros((2, 5)))
print("after call, built =", dense.built)             # الناتج: True
print("kernel shape =", dense.kernel.shape)           # الناتج: (5, 8)
print("output shape =", out.shape)                    # الناتج: (2, 8)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الوظيفي: Input ينشئ "مصفوفة رمزية" لا تحمل بيانات
inp = layers.Input(shape=(28, 28, 1), name="image")
x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
out = layers.Dense(10)(x)

model = keras.Model(inputs=inp, outputs=out)

# الشكل المتوقَّع مقابل الشكل الحقيقي
print("symbolic:", model.output_shape)                 # الناتج: (None, 10)
print("input shape:", model.input_shape)                # الناتج: (None, 28, 28, 1)

real = keras.ops.zeros((4, 28, 28, 1))
print("actual:", model(real).shape)                     # الناتج: (4, 10)
print("params:", model.count_params())                  # الناتج: 12710

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class WeightedScale(layers.Layer):
    """يضرب كلّ قناة في وزنٍ متعلَّم، ثم يضيف إزاحة متعلَّمة."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        # الأوزان تُنشأ هنا لا في __init__: الشكل صار معروفاً الآن
        self.gamma = self.add_weight(
            shape=(self.channels,),
            initializer="ones",
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
            name="beta",
        )
        super().build(input_shape)

    def call(self, inputs):
        # عمليات محايدة: تعمل على أيّ نواة
        return inputs * self.gamma + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


layer = WeightedScale(4, name="wscale")
out = layer(keras.ops.ones((3, 10, 4)))
print("output shape:", out.shape)                 # الناتج: (3, 10, 4)
print("trainable weights:", len(layer.trainable_weights))   # الناتج: 2
print("config:", layer.get_config()["channels"])   # الناتج: 4

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops

RECEIVED = {}


class MaskedMean(layers.Layer):
    """متوسّط حسابي يتجاهل الخطوات المقنّعة."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True      # إعلان: هذه الطبقة تفهم القناع

    def call(self, inputs, mask=None):
        RECEIVED["mask"] = mask
        if mask is None:
            return ops.mean(inputs, axis=1)
        m = ops.expand_dims(ops.cast(mask, inputs.dtype), axis=-1)
        total = ops.sum(inputs * m, axis=1)
        count = ops.maximum(ops.sum(m, axis=1), 1.0)
        return total / count

    def compute_mask(self, inputs, mask=None):
        return None                        # لا قناع صادر: انتهى الاستخدام هنا

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1])


# ثلاث خطوات: الثانية أصفار، أي أنها خطوة فارغة
x = keras.ops.convert_to_tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])

model = keras.Sequential([
    layers.Input(shape=(3, 2)),
    layers.Masking(mask_value=0.0),
    MaskedMean(),
])
result = model(x)
print("mask received:", keras.ops.numpy(RECEIVED["mask"]).tolist())   # الناتج: [[True, False, True]]
print("masked mean:", keras.ops.numpy(result).tolist())               # الناتج: [[1.5, 3.0]]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# 1) تسجيل الطبقة المخصّصة ليعرفها Keras عند التحميل
@keras.saving.register_keras_serializable(package="Industrial", name="WeightedScale")
class WeightedScale(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.channels,), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(
            shape=(self.channels,), initializer="zeros", trainable=True, name="beta")
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


model = keras.Sequential([
    layers.Input(shape=(6,)),
    WeightedScale(6),
    layers.Dense(2),
])
model.compile(optimizer="adam", loss="mse")
model.fit(keras.ops.zeros((8, 6)), keras.ops.zeros((8, 2)), epochs=1, verbose=0)

# 2) الحفظ الكامل: بنية + أوزان + حالة المحرّك
keras.saving.save_model(model, "sensor.keras")
reloaded = keras.saving.load_model("sensor.keras")
print("reloaded params:", reloaded.count_params())

# 3) حفظ الأوزان فقط: أخفّ، يُستعمل مع بنية معرّفة في الكود
model.save_weights("sensor.weights.h5")
model.load_weights("sensor.weights.h5")

# 4) التصدير: صيغة تنفيذية مستقلة
model.export("sensor_export", format="torch")
print("exported:", sorted(__import__("os").listdir("sensor_export")))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-20-ch20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class GluonDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        in_dim = input_shape[-1] // 2
        self.left = self.add_weight(shape=(in_dim, self.units),
                                    initializer="glorot_uniform", name="left")
        self.right = self.add_weight(shape=(in_dim, self.units),
                                     initializer="glorot_uniform", name="right")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        super().build(input_shape)

    def call(self, inputs):
        a, b = ops.split(inputs, 2, axis=-1)
        return a * ops.sigmoid(b) + self.b

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


m = keras.Sequential([layers.Input(shape=(8,)), GluonDense(4)])
m.compile(optimizer="adam", loss="mse")
x = keras.ops.random.normal(shape=(4, 8))
before = keras.ops.numpy(m(x))
keras.saving.save_model(m, "gluon.keras")
after = keras.ops.numpy(keras.saving.load_model("gluon.keras")(x))
print("max diff:", float(abs(before - after).max()))   # الناتج: 0.0
