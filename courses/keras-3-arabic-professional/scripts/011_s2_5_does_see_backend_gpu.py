# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.5 هل يرى المحرّك معالجك الرسومي؟
# المصدر  : src/chapters/ch02_setup_environment.md:150
# الترتيب : 11/13
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)
