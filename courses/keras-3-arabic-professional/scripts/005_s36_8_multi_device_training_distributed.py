# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.8 التدريب الموزّع (Multi-Device)
# المصدر  : src/chapters/ch36_acceleration.md:167
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. تعريف شبكة الأجهزة (في الإنتاج: عيّنة على جهاز بعدّة GPUs)
# devices = keras.distribution.list_devices()
# mesh = keras.distribution.DeviceMesh(
#     shape=(1, len(devices)), axis_names=("data", "model"),
#     devices=devices)
# layout = keras.distribution.TensorLayout("data", mesh=mesh)

# ─── 2. النموذج نفسه — Keras يوزّع الدفعات تلقائيًّا
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# النتيجة المتوقعة: نفس الشيفرة تعمل على جهاز واحد أو عدّة أجهزة.
