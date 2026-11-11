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
