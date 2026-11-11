# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.2 الحفظ الكامل
# المصدر  : src/chapters/ch35_saving_loading.md:33
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.
