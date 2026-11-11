# ──────────────────────────────────────────────────────────────────────────
# الفصل 26: الشبكات التكرارية المتقدّمة
# القسم   : 26.4 Attention على مُخرَج LSTM
# المصدر  : src/chapters/ch26_rnn_advanced.md:108
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

n, T, d = 500, 30, 8
x = np.random.rand(n, T, d).astype("float32")
y = (np.random.rand(n) < 0.4).astype("int32")

inp = keras.Input(shape=(T, d))
# نحتاج مُخرَج كلّ خطوة لا الحالة الأخيرة فقط
h = keras.layers.LSTM(32, return_sequences=True)(inp)
# انتباه على مُخرَجات كلّ الخطوات
attn = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(h, h)
h = keras.layers.GlobalAveragePooling1D()(attn)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.
