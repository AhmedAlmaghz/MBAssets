# ──────────────────────────────────────────────────────────────────────────
# الفصل 25: الشبكات التكرارية و LSTM و GRU
# القسم   : التحقّق العمليّ: LSTM مقابل شبكة كثيفة
# المصدر  : src/chapters/ch25_rnn_lstm.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مستشعر: 1000 تسلسل × 40 خطوة × 4 ميزات
n, T, d = 1000, 40, 4
x = np.random.randn(n, T, d).astype("float32")
# التسمية: تعتمد على متوسّط الميزة الأولى (نمط زمنيّ)
y = (x[:, :, 0].mean(axis=1) > 0.2).astype("int32")

# ─── 2. الشبكة الكثيفة (تُسطّح التسلسل)
flat = keras.Sequential([keras.Input(shape=(T, d)),
                         keras.layers.Flatten(),
                         keras.layers.Dense(32, activation="relu"),
                         keras.layers.Dense(1, activation="sigmoid")])
flat.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
flat.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
flat_acc = flat.evaluate(x[:200], y[:200], verbose=0)[1]

# ─── 3. LSTM (ترى الترتيب الزمنيّ)
rnn = keras.Sequential([keras.Input(shape=(T, d)),
                        keras.layers.LSTM(32),
                        keras.layers.Dense(1, activation="sigmoid")])
rnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
rnn.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
rnn_acc = rnn.evaluate(x[:200], y[:200], verbose=0)[1]

print(f"flat = {flat_acc:.3f}, LSTM = {rnn_acc:.3f}")
# النتيجة المتوقعة: LSTM تفوّق على المسطّحة لأنّ التسمية تعتمد على النمط الزمنيّ.
