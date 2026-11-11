# ──────────────────────────────────────────────────────────────────────────
# الفصل 25: الشبكات التكرارية و LSTM و GRU
# القسم   : LSTM مقابل GRU على نفس البيانات
# المصدر  : src/chapters/ch25_rnn_lstm.md:216
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مستشعر: 800 تسلسل × 60 خطوة × 6 ميزات
n, T, d = 800, 60, 6
x = np.random.randn(n, T, d).astype("float32")
y = (np.diff(x[:, :, 0], axis=1).max(axis=1) > 2.0).astype("int32")

# ─── 2. LSTM
lstm = keras.Sequential([keras.Input(shape=(T, d)),
                         keras.layers.LSTM(32),
                         keras.layers.Dense(1, activation="sigmoid")])
lstm.compile(optimizer="adam", loss="binary_crossentropy",
             metrics=[keras.metrics.BinaryAccuracy(name="acc")])
lstm.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

# ─── 3. GRU (25% معاملات أقلّ)
gru = keras.Sequential([keras.Input(shape=(T, d)),
                        keras.layers.GRU(32),
                        keras.layers.Dense(1, activation="sigmoid")])
gru.compile(optimizer="adam", loss="binary_crossentropy",
            metrics=[keras.metrics.BinaryAccuracy(name="acc")])
gru.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

# ─── 4. المقارنة
print(f"LSTM: acc={lstm.evaluate(x[:150], y[:150], verbose=0)[1]:.3f} "
      f"params={lstm.count_params()}")
print(f"GRU : acc={gru.evaluate(x[:150], y[:150], verbose=0)[1]:.3f} "
      f"params={gru.count_params()}")
# النتيجة المتوقعة: الدقّة متقاربة، وGRU أقلّ معاملات بـ 25%.
