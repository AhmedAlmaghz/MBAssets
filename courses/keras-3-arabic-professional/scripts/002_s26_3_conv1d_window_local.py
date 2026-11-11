# ──────────────────────────────────────────────────────────────────────────
# الفصل 26: الشبكات التكرارية المتقدّمة
# القسم   : 26.3 Conv1D: نافذة زمنية محليّة
# المصدر  : src/chapters/ch26_rnn_advanced.md:73
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

n, T, d = 500, 40, 8
x = np.random.rand(n, T, d).astype("float32")
y = (np.random.rand(n) < 0.4).astype("int32")

inp = keras.Input(shape=(T, d))
h = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inp)   # نافذة 3
h = keras.layers.MaxPooling1D(2)(h)                                       # 40 → 20
h = keras.layers.LSTM(32)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.
