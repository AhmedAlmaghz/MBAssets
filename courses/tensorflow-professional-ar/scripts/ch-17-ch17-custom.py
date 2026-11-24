#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الطبقات والنماذج المخصصة والتدريب المتقدم
Professional TensorFlow — Custom

Book slug : tensorflow-professional-ar
Chapter   : ch-17-ch17-custom (lesson 17)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-17-ch17-custom.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الطبقات والنماذج المخصصة والتدريب المتقدم
Professional TensorFlow — Custom

Book slug : tensorflow-professional-ar
Chapter   : ch-17-ch17-custom (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch17_custom.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

@keras.saving.register_keras_serializable(package="Book")
class ScaledResidual(keras.layers.Layer):
    def __init__(self, initial_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(), initializer="ones")
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.initial_scale * self.scale * inputs

    def get_config(self):
        return {**super().get_config(), "initial_scale": self.initial_scale}

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch17_custom.md
# ────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="Book")
class Scale(keras.layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(), initializer="ones")
    def call(self, inputs):
        return self.weight * inputs

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch17_custom.md
# ────────────────────────────────────────────────────────────────────────
model.save("custom.keras")
restored = keras.models.load_model("custom.keras")

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch17_custom.md
# ────────────────────────────────────────────────────────────────────────
class Trainer(keras.Model):
    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred,
                                     sample_weight=sample_weight)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-17-ch17-custom.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

@keras.saving.register_keras_serializable(package="Book")
class ScaledResidual(keras.layers.Layer):
    def __init__(self, initial_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(), initializer="ones")
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.initial_scale * self.scale * inputs

    def get_config(self):
        return {**super().get_config(), "initial_scale": self.initial_scale}

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-17-ch17-custom.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="Book")
class Scale(keras.layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(), initializer="ones")
    def call(self, inputs):
        return self.weight * inputs

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-17-ch17-custom.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
model.save("custom.keras")
restored = keras.models.load_model("custom.keras")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-17-ch17-custom.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Trainer(keras.Model):
    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred,
                                     sample_weight=sample_weight)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}
