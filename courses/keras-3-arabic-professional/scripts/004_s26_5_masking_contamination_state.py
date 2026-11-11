# ──────────────────────────────────────────────────────────────────────────
# الفصل 26: الشبكات التكرارية المتقدّمة
# القسم   : 26.5 Masking: تجنّب تلوّث الحالة
# المصدر  : src/chapters/ch26_rnn_advanced.md:141
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات بأطوال مختلفة، مُحشوّة إلى 40 خطوة
n, T, d = 500, 40, 6
x = np.random.rand(n, T, d).astype("float32")
# نُصفّر الخطوات الأخيرة في بعض التسلسلات (محاكاة تسلسلات قصيرة)
lengths = np.random.randint(10, T, size=(n,))
for i, L in enumerate(lengths):
    x[i, L:, :] = 0.0
y = (np.random.rand(n) < 0.4).astype("int32")

# ─── 2. Masking كأول طبقة
inp = keras.Input(shape=(T, d))
h = keras.layers.Masking(mask_value=0.0)(inp)
h = keras.layers.LSTM(32)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: أداء أفضل مقارنة بـ LSTM بلا Masking
# لأنّ الحالة لم تتلوّث بالأصفار.
