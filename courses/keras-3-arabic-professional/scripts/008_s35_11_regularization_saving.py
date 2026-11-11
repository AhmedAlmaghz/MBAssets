# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.11 تنظيم نقاط الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:259
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.
