#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات الأساسية: Dense والتنشيط والتسطيح
Keras 3 in Arabic — Core Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-06-ch06-core-layers (lesson 6)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-06-ch06-core-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات الأساسية: Dense والتنشيط والتسطيح
Keras 3 in Arabic — Core Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-06-ch06-core-layers (lesson 6)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-06-ch06-core-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات الأساسية: Dense والتنشيط والتسطيح
Keras 3 in Arabic — Core Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-06-ch06-core-layers (lesson 6)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch06_core_layers/001_s6_2_input_inputlayer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.2 `Input` و`InputLayer`
# المصدر  : src/chapters/ch06_core_layers.md:42
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch06_core_layers/002_flatten.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Flatten`
# المصدر  : src/chapters/ch06_core_layers.md:68
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch06_core_layers/003_reshape.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Reshape`
# المصدر  : src/chapters/ch06_core_layers.md:77
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch06_core_layers/004_globalaveragepooling1d.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `GlobalAveragePooling1D`
# المصدر  : src/chapters/ch06_core_layers.md:86
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch06_core_layers/005_s6_4_functions_activation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.4 دوال التنشيط كطبقات
# المصدر  : src/chapters/ch06_core_layers.md:104
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch06_core_layers/006_spatialdropout_error.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `SpatialDropout`: الخطأ الشائع
# المصدر  : src/chapters/ch06_core_layers.md:149
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch06_core_layers/007_s6_6_masking.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.6 `Masking`: تعليم الشبكة ما يهمّ
# المصدر  : src/chapters/ch06_core_layers.md:171
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch06_core_layers/008_s6_7_layers_custom_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.7 الطبقات المخصّصة: الأساس
# المصدر  : src/chapters/ch06_core_layers.md:181
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch06_core_layers/009_s6_9_mlp_example_solved_layers.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.9 مثال محلول: شبكة MLP بثلاث طبقات
# المصدر  : src/chapters/ch06_core_layers.md:230
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.2 `Input` و`InputLayer`
# المصدر  : src/chapters/ch06_core_layers.md:42
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Flatten`
# المصدر  : src/chapters/ch06_core_layers.md:68
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Reshape`
# المصدر  : src/chapters/ch06_core_layers.md:77
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `GlobalAveragePooling1D`
# المصدر  : src/chapters/ch06_core_layers.md:86
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.4 دوال التنشيط كطبقات
# المصدر  : src/chapters/ch06_core_layers.md:104
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `SpatialDropout`: الخطأ الشائع
# المصدر  : src/chapters/ch06_core_layers.md:149
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.6 `Masking`: تعليم الشبكة ما يهمّ
# المصدر  : src/chapters/ch06_core_layers.md:171
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.7 الطبقات المخصّصة: الأساس
# المصدر  : src/chapters/ch06_core_layers.md:181
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.9 مثال محلول: شبكة MLP بثلاث طبقات
# المصدر  : src/chapters/ch06_core_layers.md:230
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات الأساسية: Dense والتنشيط والتسطيح
Keras 3 in Arabic — Core Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-06-ch06-core-layers (lesson 6)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch06_core_layers/001_s6_2_input_inputlayer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.2 `Input` و`InputLayer`
# المصدر  : src/chapters/ch06_core_layers.md:42
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch06_core_layers/002_flatten.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Flatten`
# المصدر  : src/chapters/ch06_core_layers.md:68
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch06_core_layers/003_reshape.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Reshape`
# المصدر  : src/chapters/ch06_core_layers.md:77
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch06_core_layers/004_globalaveragepooling1d.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `GlobalAveragePooling1D`
# المصدر  : src/chapters/ch06_core_layers.md:86
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch06_core_layers/005_s6_4_functions_activation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.4 دوال التنشيط كطبقات
# المصدر  : src/chapters/ch06_core_layers.md:104
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch06_core_layers/006_spatialdropout_error.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `SpatialDropout`: الخطأ الشائع
# المصدر  : src/chapters/ch06_core_layers.md:149
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch06_core_layers/007_s6_6_masking.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.6 `Masking`: تعليم الشبكة ما يهمّ
# المصدر  : src/chapters/ch06_core_layers.md:171
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch06_core_layers/008_s6_7_layers_custom_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.7 الطبقات المخصّصة: الأساس
# المصدر  : src/chapters/ch06_core_layers.md:181
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch06_core_layers/009_s6_9_mlp_example_solved_layers.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.9 مثال محلول: شبكة MLP بثلاث طبقات
# المصدر  : src/chapters/ch06_core_layers.md:230
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch06_core_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.2 `Input` و`InputLayer`
# المصدر  : src/chapters/ch06_core_layers.md:42
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Flatten`
# المصدر  : src/chapters/ch06_core_layers.md:68
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `Reshape`
# المصدر  : src/chapters/ch06_core_layers.md:77
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `GlobalAveragePooling1D`
# المصدر  : src/chapters/ch06_core_layers.md:86
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.4 دوال التنشيط كطبقات
# المصدر  : src/chapters/ch06_core_layers.md:104
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : `SpatialDropout`: الخطأ الشائع
# المصدر  : src/chapters/ch06_core_layers.md:149
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.6 `Masking`: تعليم الشبكة ما يهمّ
# المصدر  : src/chapters/ch06_core_layers.md:171
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.7 الطبقات المخصّصة: الأساس
# المصدر  : src/chapters/ch06_core_layers.md:181
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.9 مثال محلول: شبكة MLP بثلاث طبقات
# المصدر  : src/chapters/ch06_core_layers.md:230
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Flatten()(keras.Input((28, 28)))
print(x.shape)   # (None, 784)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Reshape((7, 7, 4))(keras.Input((196,)))
print(x.shape)   # (None, 7, 7, 4)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.GlobalAveragePooling1D()(keras.Input((100, 64)))
print(x.shape)   # (None, 64)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المختصرة (مفضّلة)
x = keras.layers.Dense(64, activation="relu")(x)

# الطريقة الصريحة
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation("relu")(x)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# أبعاد مربّعة (4D)
x = keras.layers.SpatialDropout2D(0.2)(x)

# أبعاد تسلسلية (3D)
x = keras.layers.SpatialDropout1D(0.2)(x)

# أبعاد مكانية (5D)
x = keras.layers.SpatialDropout3D(0.2)(x)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
x = keras.layers.Masking(mask_value=0.0)(keras.Input((100, 64)))

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras

class Scale(keras.layers.Layer):
    """يضرب الخرج في معامل قابل للتعلّم."""

    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "init_scale": self.init_scale}

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-06-ch06-core-layers.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()
