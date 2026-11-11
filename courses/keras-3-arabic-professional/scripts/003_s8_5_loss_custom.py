# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.5 خسارة مخصّصة
# المصدر  : src/chapters/ch08_loss_functions.md:139
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
