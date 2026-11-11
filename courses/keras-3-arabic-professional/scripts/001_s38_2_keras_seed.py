# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.2 البذرة في Keras 3
# المصدر  : src/chapters/ch38_reproducibility.md:30
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.
