# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب البارامترات يدويًا والتحقّق
# المصدر  : src/chapters/ch18_convolution_math.md:337
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True
