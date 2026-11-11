# ──────────────────────────────────────────────────────────────────────────
# الفصل 25: الشبكات التكرارية و LSTM و GRU
# القسم   : 25.6 التنفيذ في Keras
# المصدر  : src/chapters/ch25_rnn_lstm.md:121
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 600 تسلسل، كلّ واحد 40 خطوة، كلّ خطوة 8 ميزات
n, T, d = 600, 40, 8
x = np.random.rand(n, T, d).astype("float32")
y = (np.random.rand(n) < 0.3).astype("int32")

# ─── 2. نموذج LSTM
inp = keras.Input(shape=(T, d))
h = keras.layers.LSTM(32, return_sequences=False)(inp)   # الحالة الأخيرة فقط
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
)
history = model.fit(x, y, epochs=10, batch_size=32,
                    validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة ≈ 0.55–0.65 على بيانات عشوائية.
