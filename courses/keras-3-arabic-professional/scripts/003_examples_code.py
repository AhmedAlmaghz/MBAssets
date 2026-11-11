# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:201
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True
