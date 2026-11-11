# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 5 — ميزات بمقياس مختلف
# المصدر  : src/chapters/ch40_common_errors.md:119
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.
