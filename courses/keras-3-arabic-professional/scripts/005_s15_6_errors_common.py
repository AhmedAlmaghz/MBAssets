# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.6 أخطاء شائعة في هذا الفصل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:249
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

norm = keras.layers.Normalization()
x = np.array([[[0.0, 100.0, 250.0]]], dtype="float32")
print("قبل:", norm(x, training=True).numpy().ravel())  # النتيجة المتوقعة: قيم قريبة من 0
print("بعد:", norm(x, training=False).numpy().ravel()) # بلا أثر قبل التعلّم
