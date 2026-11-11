# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.12 مثال: حفظ الأوزان لنقلها بين بنيتين
# المصدر  : src/chapters/ch35_saving_loading.md:291
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.
