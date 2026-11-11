# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.4 نقطة الحفظ (ModelCheckpoint)
# المصدر  : src/chapters/ch35_saving_loading.md:84
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.
