#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات والنماذج المخصّصة
Keras 3 in Arabic — Custom Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-34-ch34-custom-layers (lesson 34)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-34-ch34-custom-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات والنماذج المخصّصة
Keras 3 in Arabic — Custom Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-34-ch34-custom-layers (lesson 34)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-34-ch34-custom-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات والنماذج المخصّصة
Keras 3 in Arabic — Custom Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-34-ch34-custom-layers (lesson 34)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch34_custom_layers/001_s34_2_layer_custom_simple.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.2 طبقة مخصّصة بسيطة
# المصدر  : src/chapters/ch34_custom_layers.md:32
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch34_custom_layers/002_s34_3_model_training_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.3 نموذج مخصّص بخطوة تدريب مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:83
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch34_custom_layers/003_s34_4_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.4 الحفظ والتحميل
# المصدر  : src/chapters/ch34_custom_layers.md:133
# الترتيب : 3/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch34_custom_layers/004_s34_5_example_solved_layer_attention.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.5 مثال محلول: طبقة انتباه مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch34_custom_layers/005_s34_7_masking_layers_mask.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.7 الطبقات التي تُدعم القناع (Masking)
# المصدر  : src/chapters/ch34_custom_layers.md:221
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch34_custom_layers/006_s34_8_compute_shape.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.8 حساب شكل المُخرَج
# المصدر  : src/chapters/ch34_custom_layers.md:262
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch34_custom_layers/007_s34_9_test_layer_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.9 اختبار الطبقة المخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:290
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.2 طبقة مخصّصة بسيطة
# المصدر  : src/chapters/ch34_custom_layers.md:32
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.3 نموذج مخصّص بخطوة تدريب مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:83
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.4 الحفظ والتحميل
# المصدر  : src/chapters/ch34_custom_layers.md:133
# الترتيب : 3/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.5 مثال محلول: طبقة انتباه مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.7 الطبقات التي تُدعم القناع (Masking)
# المصدر  : src/chapters/ch34_custom_layers.md:221
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.8 حساب شكل المُخرَج
# المصدر  : src/chapters/ch34_custom_layers.md:262
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.9 اختبار الطبقة المخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:290
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات والنماذج المخصّصة
Keras 3 in Arabic — Custom Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-34-ch34-custom-layers (lesson 34)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch34_custom_layers/001_s34_2_layer_custom_simple.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.2 طبقة مخصّصة بسيطة
# المصدر  : src/chapters/ch34_custom_layers.md:32
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch34_custom_layers/002_s34_3_model_training_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.3 نموذج مخصّص بخطوة تدريب مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:83
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch34_custom_layers/003_s34_4_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.4 الحفظ والتحميل
# المصدر  : src/chapters/ch34_custom_layers.md:133
# الترتيب : 3/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch34_custom_layers/004_s34_5_example_solved_layer_attention.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.5 مثال محلول: طبقة انتباه مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch34_custom_layers/005_s34_7_masking_layers_mask.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.7 الطبقات التي تُدعم القناع (Masking)
# المصدر  : src/chapters/ch34_custom_layers.md:221
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch34_custom_layers/006_s34_8_compute_shape.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.8 حساب شكل المُخرَج
# المصدر  : src/chapters/ch34_custom_layers.md:262
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch34_custom_layers/007_s34_9_test_layer_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.9 اختبار الطبقة المخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:290
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch34_custom_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.2 طبقة مخصّصة بسيطة
# المصدر  : src/chapters/ch34_custom_layers.md:32
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.3 نموذج مخصّص بخطوة تدريب مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:83
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.4 الحفظ والتحميل
# المصدر  : src/chapters/ch34_custom_layers.md:133
# الترتيب : 3/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.5 مثال محلول: طبقة انتباه مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.7 الطبقات التي تُدعم القناع (Masking)
# المصدر  : src/chapters/ch34_custom_layers.md:221
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.8 حساب شكل المُخرَج
# المصدر  : src/chapters/ch34_custom_layers.md:262
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.9 اختبار الطبقة المخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:290
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

@keras.saving.register_keras_serializable(package="book")
class MyDense(keras.layers.Layer):
    """طبقة خطّيّة مخصّصة: out = activation(W @ x + b)."""

    def __init__(self, units, activation=None, **kwargs):
        # نُخزّن الإعدادات فقط — لا أوزان هنا
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # يُنشئ الأوزان مرة واحدة، بحسب شكل المُدخل
        last_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)   # يُخبر Keras أنّ الطبقة بُنيت

    def call(self, inputs):
        # المسار الأماميّ
        return self.activation(inputs @ self.w + self.b)

    def get_config(self):
        # يُصفّ الطبقة للحفظ
        config = super().get_config()
        config.update({"units": self.units,
                       "activation": keras.activations.serialize(self.activation)})
        return config

# استخدامها
layer = MyDense(32, activation="relu")
out = layer(keras.ops.ones((4, 16)))
# النتيجة المتوقعة: شكل (4, 32).

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

class AttentionPooling(keras.layers.Layer):
    """تجميع موزون على تسلسل: (batch, T, d) → (batch, d)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # وزن واحد لكلّ بُعد ميزة
        self.w = self.add_weight(
            name="attn_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # مُعاملات الانتباه: (batch, T)
        scores = keras.ops.sum(inputs * self.w, axis=-1)
        attn = keras.ops.softmax(scores, axis=-1)
        # التجميع: (batch, d)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها
x = keras.ops.ones((4, 10, 16))
out = AttentionPooling()(x)
# النتيجة المتوقعة: شكل (4, 16).

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

class MaskedPooling(keras.layers.Layer):
    """تجميع موزون يتجاهل الخطوات المُموّهة."""

    def build(self, input_shape):
        self.w = self.add_weight(
            name="pool_weights",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.supports_masking = True      # يُخبر Keras أنّ الطبقة تُدعم القناع

    def call(self, inputs, mask=None):
        scores = keras.ops.sum(inputs * self.w, axis=-1)   # (B, T)
        if mask is not None:
            # القناع: False للخطوات المُموّهة — نُخفيها
            scores = keras.ops.where(mask, scores,
                                     keras.ops.full(keras.ops.shape(scores), -1e9))
        attn = keras.ops.softmax(scores, axis=-1)
        return keras.ops.sum(inputs * attn[..., None], axis=1)

    def get_config(self):
        return super().get_config()

# استخدامها بعد Masking
inp = keras.Input(shape=(20, 16))
x = keras.layers.Masking(mask_value=0.0)(inp)
out = MaskedPooling()(x)
model = keras.Model(inp, out)
# النتيجة المتوقعة: التجميع يتجاهل الخطوات المُحشوّة بالأصفار.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

class MyReshape(keras.layers.Layer):
    """إعادة تشكيل بحجم مُحدَّد."""

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        return keras.ops.reshape(inputs, (-1,) + self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        return {**super().get_config(), "target_shape": self.target_shape}

layer = MyReshape((4, 4))
out = layer(keras.ops.ones((2, 16)))
# النتيجة المتوقعة: شكل (2, 4, 4).

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-34-ch34-custom-layers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def test_shape():
    """الطبقة تُعطي شكلًا صحيحًا."""
    layer = MyDense(32, activation="relu")
    out = layer(keras.ops.ones((4, 16)))
    assert out.shape == (4, 32)

def test_gradient_flows():
    """التدرّج يمرّ عبر الأوزان."""
    layer = MyDense(8)
    x = keras.ops.ones((4, 16))
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    with tf.GradientTape() as tape:
        layer.build(x.shape)
        y = layer(x)
        loss = keras.ops.mean(keras.ops.square(y))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None for g in grads)

def test_save_load():
    """الطبقة تُحفَظ وتُحمَّل."""
    model = keras.Sequential([keras.Input(shape=(16,)), MyDense(8)])
    model.save("test_layer.keras")
    m2 = keras.saving.load_model("test_layer.keras",
                                 custom_objects={"MyDense": MyDense})
    n_custom = sum(isinstance(l, MyDense) for l in m2.layers)
    assert n_custom == 1

test_shape()
test_gradient_flows()
test_save_load()
# النتيجة المتوقعة: الاختبارات الثلاثة تنجح.
