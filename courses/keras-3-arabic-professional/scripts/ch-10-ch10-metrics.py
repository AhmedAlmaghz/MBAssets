#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
Keras 3 in Arabic — Metrics

Book slug : keras-3-arabic-professional
Chapter   : ch-10-ch10-metrics (lesson 10)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-10-ch10-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
Keras 3 in Arabic — Metrics

Book slug : keras-3-arabic-professional
Chapter   : ch-10-ch10-metrics (lesson 10)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-10-ch10-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
Keras 3 in Arabic — Metrics

Book slug : keras-3-arabic-professional
Chapter   : ch-10-ch10-metrics (lesson 10)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch10_metrics/001_s10_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:81
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch10_metrics/002_s10_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:114
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch10_metrics/003_scale.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch10_metrics/004_s10_5_example_solved_system_fault.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:210
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch10_metrics/005_s10_5_example_solved_system_fault.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:81
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:114
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:210
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
Keras 3 in Arabic — Metrics

Book slug : keras-3-arabic-professional
Chapter   : ch-10-ch10-metrics (lesson 10)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch10_metrics/001_s10_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:81
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch10_metrics/002_s10_3_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:114
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch10_metrics/003_scale.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch10_metrics/004_s10_5_example_solved_system_fault.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:210
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch10_metrics/005_s10_5_example_solved_system_fault.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch10_metrics.md
# ────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:81
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:114
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:210
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.5 مثال محلول: نظام اكتشاف عطل في خط إنتاج
# المصدر  : src/chapters/ch10_metrics.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ── بيانات تجريبية صغيرة: 6 عيّنات، positives نادرة
y_true = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
y_pred = np.array([0.1, 0.8, 0.9, 0.4, 0.3, 0.2], dtype="float32")

# ── مقاييس Keras الجاهزة
acc = keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_true, y_pred)
print("accuracy:", float(acc.result()))   # النتيجة المتوقعة: 0.6666666865348816

prec = keras.metrics.Precision()
rec = keras.metrics.Recall()
f1 = keras.metrics.F1Score(average="micro")
for m in (prec, rec):
    m.update_state(y_true, y_pred)
# F1Score يطلب شكل (batch, output_dim) لا (batch,)
f1.update_state(y_true[:, None], y_pred[:, None])
print("precision:", float(prec.result()))  # النتيجة المتوقعة: 0.5
print("recall:", float(rec.result()))     # النتيجة المتوقعة: 0.6666666865348816
print("f1:", float(f1.result()))          # النتيجة المتوقعة: 0.5714285969734192

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# محاكاة بسيطة: احتمالات التنبؤ لعتبات مختلفة
scores = np.array([0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype="float32")

for thr in (0.3, 0.5, 0.7, 0.8):
    m_f2 = keras.metrics.FBetaScore(beta=2.0, threshold=thr)
    # F1/F-beta يطلبان شكل (batch, output_dim)
    m_f2.update_state(labels[:, None], scores[:, None])
    m_p = keras.metrics.Precision(thresholds=thr)
    m_p.update_state(labels, scores)
    print(f"thr={thr}  F2={float(m_f2.result()):.3f}  P={float(m_p.result()):.3f}")
# النتيجة المتوقعة: العتبة 0.7 ترفع Precision مع إبقاء Recall مقبولًا

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-10-ch10-metrics.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
m_pr = keras.metrics.AUC(curve="PR")
m_pr.update_state(labels, scores)
print("AUC-PR:", float(m_pr.result()))
