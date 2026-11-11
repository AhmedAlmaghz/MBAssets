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
