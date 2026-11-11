# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : الجداول في Keras 3
# المصدر  : src/chapters/ch09_optimizers.md:132
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

# أُسّي
expo = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False,
)

# جيبي
cosine = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.0,          # أقلّ قيمة نهائية كنسبة من الابتدائي
    warmup_steps=500,
)

# متعدّد الطور
piece = PiecewiseConstantDecay(
    boundaries=[500, 1000, 2000],
    values=[1e-3, 5e-4, 1e-4, 1e-5],
)
