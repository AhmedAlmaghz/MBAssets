# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `CategoryEncoding` و`Discretization`
# المصدر  : src/chapters/ch16_text_layers.md:219
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة
