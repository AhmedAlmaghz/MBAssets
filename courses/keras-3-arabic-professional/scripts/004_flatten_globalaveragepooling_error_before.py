# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 2 — Flatten قبل GlobalAveragePooling
# المصدر  : src/chapters/ch40_common_errors.md:61
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).
