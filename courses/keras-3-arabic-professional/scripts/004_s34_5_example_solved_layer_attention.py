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
