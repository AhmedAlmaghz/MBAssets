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
