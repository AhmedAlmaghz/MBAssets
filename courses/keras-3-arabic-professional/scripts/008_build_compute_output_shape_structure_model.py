# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : بنية النموذج: `build` و`compute_output_shape`
# المصدر  : src/chapters/ch04_model_architecture.md:170
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config
