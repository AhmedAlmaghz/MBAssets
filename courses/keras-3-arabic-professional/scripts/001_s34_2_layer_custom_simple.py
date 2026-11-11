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
