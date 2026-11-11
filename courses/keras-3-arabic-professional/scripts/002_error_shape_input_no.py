# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 1 — شكل مُدخل لا يطابق
# المصدر  : src/chapters/ch40_common_errors.md:39
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.
