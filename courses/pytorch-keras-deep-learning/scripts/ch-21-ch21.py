#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Keras 3 فوق نواة PyTorch
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 21 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-21-ch21 (lesson 21)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-21-ch21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — Keras 3 فوق نواة PyTorch
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 21 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-21-ch21 (lesson 21)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # يُقرأ قبل استيراد keras

import keras
from keras import ops

print(keras.backend.backend())      # torch
print(keras.backend.standardize_dtype("float32"))

a = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
b = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
print(ops.matmul(a, b))
# الناتج: tensor([[2., 1.],
#                 [4., 3.]], device='cuda:0', dtype=torch.float32)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الوظيفي: explicit graph, أفضل للتصدير
inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)
x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(model.output_shape)     # (None, 10)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import keras
from keras import layers


class TorchBlock(layers.Layer):
    """غلاف لطبقة PyTorch داخل Keras — للعملية فقط، لا للضبط."""

    def __init__(self, torch_module, **kwargs):
        super().__init__(**kwargs)
        self.torch_module = torch_module

    def call(self, inputs, training=False):
        t = ops.convert_to_tensor(inputs)
        out = self.torch_module(t)
        return ops.convert_to_tensor(out.detach().cpu().numpy())

    def get_config(self):
        return super().get_config()


from keras import ops


def torch_to_keras(torch_model):
    """يحوّل نموذج تصنيف بسيط مبني بـ torch.nn إلى keras.Model."""
    x = keras.Input(shape=(3, 32, 32))
    y = ops.convert_to_tensor(x)
    y = torch_model.features(y)
    y = torch_model.classifier(torch.flatten(y, 1))
    return keras.Model(x, y, name="converted")

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class ContrastiveModel(keras.Model):
    """نموذج تصنيف + رأس تباين، مع حلقة تدريب مكتوبة بأنماط Keras."""

    def __init__(self, num_classes=10, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.features = layers.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Rescaling(1 / 255.0),
            layers.Conv2D(32, 3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
        ])
        self.projector = layers.Dense(projection_dim)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        h = self.features(x, training=training)
        return self.classifier(h)

    def train_step(self, data):
        """حلقة تدريب مخصّصة على نواة PyTorch."""
        x, y = data
        y_pred = self(x, training=True)
        loss = self.compute_loss(x, y, y_pred)

        # حساب التدرّجات على نواة torch
        self.zero_grad()
        loss.backward()
        grads = [v.value.grad for v in self.trainable_weights]
        self.optimizer.apply(grads, self.trainable_weights)

        return self.compute_metrics(x, y, y_pred)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import keras

# نطاق جهاز صريح
with keras.device("CPU"):
    small = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation="relu"),
    ])

# نطاق منطقي: يبقى الجهاز الافتراضي
with keras.device("default"):
    same = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation="relu"),
    ])

print(small.weights[0].shape)
# الناتج: (64, 32)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch21.html
# ────────────────────────────────────────────────────────────────────────
import keras

def check_backend(path):
    print("backend:", keras.backend.backend())
    model = keras.models.load_model(path)
    print("input shape:", model.input_shape)
    return model

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # يُقرأ قبل استيراد keras

import keras
from keras import ops

print(keras.backend.backend())      # torch
print(keras.backend.standardize_dtype("float32"))

a = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
b = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
print(ops.matmul(a, b))
# الناتج: tensor([[2., 1.],
#                 [4., 3.]], device='cuda:0', dtype=torch.float32)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# النمط الوظيفي: explicit graph, أفضل للتصدير
inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)
x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(model.output_shape)     # (None, 10)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import keras
from keras import layers


class TorchBlock(layers.Layer):
    """غلاف لطبقة PyTorch داخل Keras — للعملية فقط، لا للضبط."""

    def __init__(self, torch_module, **kwargs):
        super().__init__(**kwargs)
        self.torch_module = torch_module

    def call(self, inputs, training=False):
        t = ops.convert_to_tensor(inputs)
        out = self.torch_module(t)
        return ops.convert_to_tensor(out.detach().cpu().numpy())

    def get_config(self):
        return super().get_config()


from keras import ops


def torch_to_keras(torch_model):
    """يحوّل نموذج تصنيف بسيط مبني بـ torch.nn إلى keras.Model."""
    x = keras.Input(shape=(3, 32, 32))
    y = ops.convert_to_tensor(x)
    y = torch_model.features(y)
    y = torch_model.classifier(torch.flatten(y, 1))
    return keras.Model(x, y, name="converted")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers, ops


class ContrastiveModel(keras.Model):
    """نموذج تصنيف + رأس تباين، مع حلقة تدريب مكتوبة بأنماط Keras."""

    def __init__(self, num_classes=10, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.features = layers.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Rescaling(1 / 255.0),
            layers.Conv2D(32, 3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
        ])
        self.projector = layers.Dense(projection_dim)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        h = self.features(x, training=training)
        return self.classifier(h)

    def train_step(self, data):
        """حلقة تدريب مخصّصة على نواة PyTorch."""
        x, y = data
        y_pred = self(x, training=True)
        loss = self.compute_loss(x, y, y_pred)

        # حساب التدرّجات على نواة torch
        self.zero_grad()
        loss.backward()
        grads = [v.value.grad for v in self.trainable_weights]
        self.optimizer.apply(grads, self.trainable_weights)

        return self.compute_metrics(x, y, y_pred)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

# نطاق جهاز صريح
with keras.device("CPU"):
    small = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation="relu"),
    ])

# نطاق منطقي: يبقى الجهاز الافتراضي
with keras.device("default"):
    same = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation="relu"),
    ])

print(small.weights[0].shape)
# الناتج: (64, 32)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-21-ch21.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

def check_backend(path):
    print("backend:", keras.backend.backend())
    model = keras.models.load_model(path)
    print("input shape:", model.input_shape)
    return model
