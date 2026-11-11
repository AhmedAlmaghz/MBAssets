# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فخّ مشاركة الأوزان
# المصدر  : src/chapters/ch04_model_architecture.md:115
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط
