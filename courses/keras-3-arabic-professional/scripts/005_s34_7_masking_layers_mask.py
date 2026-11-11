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
