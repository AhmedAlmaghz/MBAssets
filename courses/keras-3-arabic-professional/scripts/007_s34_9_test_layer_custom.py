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
