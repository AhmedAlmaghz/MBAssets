# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.2 تفعيل الدقّة المختلطة
# المصدر  : src/chapters/ch36_acceleration.md:28
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. ضبط الدقّة المختلطة قبل بناء النموذج
keras.mixed_precision.set_global_policy("mixed_float16")

# ─── 2. بناء النموذج (آخر طبقة تحتفظ بـ float32)
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(10)(x)    # بلا تنشيط — لوجيتس
model = keras.Model(inp, out)

# ─── 3. الخسارة من اللوجيتس (لا softmax)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ─── 4. إعادة ضبط بعد التدريب (اختياريّ)
# keras.mixed_precision.set_global_policy("float32")
# النتيجة المتوقعة: تدريب أسرع بـ 1.5–2× على GPU.
