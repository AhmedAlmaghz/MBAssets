# ──────────────────────────────────────────────────────────────────────────
# الفصل 26: الشبكات التكرارية المتقدّمة
# القسم   : 26.2 Bidirectional: قراءة في الاتجاهين
# المصدر  : src/chapters/ch26_rnn_advanced.md:41
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 500 تسلسل، 30 خطوة، 6 ميزات
n, T, d = 500, 30, 6
x = np.random.rand(n, T, d).astype("float32")
y = np.random.randint(0, 2, size=(n,)).astype("int32")

# ─── 2. نموذج ثنائيّ الاتجاه
inp = keras.Input(shape=(T, d))
# Bidirectional يُنشئ نسختين من LSTM ويُدمج مُخرَجاتهما (الافتراضيّ: دمج)
h = keras.layers.Bidirectional(keras.layers.LSTM(32))(inp)
# حجم المُخرَج = 32 × 2 = 64
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.
