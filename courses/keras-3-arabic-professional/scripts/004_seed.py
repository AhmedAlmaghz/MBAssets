# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : البذرة: كيف تجعل الشبكة نفسها في كل مرة؟
# المصدر  : src/chapters/ch11_initializers.md:282
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")
