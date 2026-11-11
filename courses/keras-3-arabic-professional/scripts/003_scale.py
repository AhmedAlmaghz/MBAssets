# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : مقياس مخصّص
# المصدر  : src/chapters/ch10_metrics.md:127
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np


class RecallAtK(keras.metrics.Metric):
    """يقيس نسبة الاستدعاء من بين top-k تنبّؤات."""

    def __init__(self, k=3, name="recall_at_k"):
        super().__init__(name=name)
        self.k = k
        # عدّادات حالة تُحدَّث عبر الدفعات وتُحفظ في checkpoints
        self.hits = self.add_weight(name="hits", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # top-k indices لكل عيّنة
        top_k_idx = keras.ops.argsort(y_pred, axis=-1)[:, -self.k:]
        # هل ظهرت الإجابة الصحيحة ضمن أعلى k؟ (الـone-hot OneHot)
        true_labels = keras.ops.argmax(y_true, axis=-1, keepdims=True)
        hit = keras.ops.any(keras.ops.equal(top_k_idx, true_labels), axis=-1)
        hit = keras.ops.cast(hit, self.dtype)
        if sample_weight is not None:
            hit = hit * sample_weight
        self.hits.assign_add(keras.ops.sum(hit))
        self.total.assign_add(keras.ops.sum(keras.ops.cast(
            keras.ops.ones_like(hit), self.dtype)))

    def result(self):
        return self.hits / (self.total + keras.backend.epsilon())

    def reset_state(self):
        self.hits.assign(0)
        self.total.assign(0)


m = RecallAtK(k=2)
m.update_state(np.eye(3, dtype="float32"), np.eye(3, dtype="float32"))
print("recall@2:", float(m.result()))  # النتيجة المتوقعة: 1.0
m.reset_state()
print("after reset:", float(m.result()))  # النتيجة المتوقعة: 0.0
