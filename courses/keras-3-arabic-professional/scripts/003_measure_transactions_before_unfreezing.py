# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : قياس المعاملات القابلة للتدريب قبل وبعد الفكّ
# المصدر  : src/chapters/ch21_transfer_learning.md:131
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. بنية قاعدة صغيرة (محاكاة backbone مُدرَّب مسبقًا)
base = keras.Sequential([keras.Input(shape=(64, 64, 3)),
                         keras.layers.Conv2D(16, 3, padding="same",
                                             use_bias=False),
                         keras.layers.Conv2D(32, 3, padding="same",
                                             use_bias=False)])

def count_trainable(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))

import numpy as np

# ─── 2. مجمّدة
base.trainable = False
frozen = count_trainable(base)
# ─── 3. مفكوكة
base.trainable = True
unfrozen = count_trainable(base)
print(f"frozen = {frozen}, unfrozen = {unfrozen}")

# ─── 4. فكّ جزئيّ: الطبقة الأخيرة فقط
base.trainable = False
base.layers[-1].trainable = True
partial = count_trainable(base)
print(f"partial (last layer) = {partial}")
# النتيجة المتوقعة: frozen = 0، unfrozen = كلّ الأوزان،
# partial بينهما — وهذا هو أساس التدريب الدقيق الجزئيّ.
