# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `.keras` مقابل `.weights.h5`: فرق جوهري
# المصدر  : src/chapters/ch12_callbacks.md:218
# الترتيب : 4/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح
