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
