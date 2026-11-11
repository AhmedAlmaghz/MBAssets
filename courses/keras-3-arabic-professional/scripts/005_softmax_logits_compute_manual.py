# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : حساب softmax يدويًّا وربطه بـ logits
# المصدر  : src/chapters/ch07_activations.md:247
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
from keras import ops

# ثلاثة logits: نظام كشف الاحتيال (عادي / مشبوه / احتيال)
logits = ops.array([2.0, 1.0, 0.1])

def stable_softmax(z, axis=-1):
    """softmax مع طرح أكبر قيمة قبل الأس (حيلة log-sum-exp)."""
    m = ops.max(z, axis=axis, keepdims=True)
    e = ops.exp(z - m)
    return e / ops.sum(e, axis=axis, keepdims=True)

p = stable_softmax(logits)
print(p)
print("المجموع:", float(ops.sum(p)))
print("الفئة المتوقعة:", int(ops.argmax(p)))
# النتيجة المتوقعة: [0.659 0.242 0.099] ، المجموع: 1.0 ، الفئة: 0

# النسخة التي يفعلها Keras داخليًا مع from_logits=True: اللوغاريتمات مباشرة
m = ops.max(logits)
log_p = logits - m - ops.log(ops.sum(ops.exp(logits - m)))
print(log_p)
# النتيجة المتوقعة: [-0.417  -0.835  -1.417]

# قارن بـ softmax غير المستقر: لولا الطرح لانفجر الأسّ
big = ops.array([1000.0, 1001.0, 1002.0])
print(stable_softmax(big))   # النتيجة المتوقعة: [0.090 0.245 0.665] بدون overflow
