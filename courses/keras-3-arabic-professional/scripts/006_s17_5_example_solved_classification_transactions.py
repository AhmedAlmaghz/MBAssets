# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:261
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

counts = np.array([98000, 2000])           # 98% / 2%
n = counts.sum()
weights = n / (len(counts) * counts)       # [0.510, 25.0]
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(class_weight)   # النتيجة المتوقعة: {0: 0.5102, 1: 25.0}

model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)
