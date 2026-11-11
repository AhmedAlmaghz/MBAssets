# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.7 استئناف التدريب
# المصدر  : src/chapters/ch35_saving_loading.md:160
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.
