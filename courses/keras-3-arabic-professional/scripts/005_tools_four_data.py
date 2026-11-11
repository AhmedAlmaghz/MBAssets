# ──────────────────────────────────────────────────────────────────────────
# الفصل 26: الشبكات التكرارية المتقدّمة
# القسم   : مقارنة الأدوات الأربع على نفس البيانات
# المصدر  : src/chapters/ch26_rnn_advanced.md:201
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات بأطوال مختلفة (مُحشوّة)
n, T, d = 600, 30, 6
x = np.random.randn(n, T, d).astype("float32")
lengths = np.random.randint(10, T, size=(n,))
for i, L in enumerate(lengths):
    x[i, L:, :] = 0.0
y = (x[:, :, 0].mean(axis=1) > 0.1).astype("int32")

def make_model(with_mask=False, bidir=False, conv=False):
    """يبني نموذجًا بأدوات محدّدة."""
    layers = [keras.Input(shape=(T, d))]
    if with_mask:
        layers.append(keras.layers.Masking(mask_value=0.0))
    if conv:
        layers += [keras.layers.Conv1D(16, 3, padding="same",
                                       activation="relu"),
                   keras.layers.MaxPooling1D(2)]
    core = keras.layers.LSTM(32)
    layers.append(keras.layers.Bidirectional(core) if bidir else core)
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    m = keras.Sequential(layers)
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    return m

configs = [
    ("LSTM فقط", dict()),
    ("+ Masking", dict(with_mask=True)),
    ("+ Masking + Bidir", dict(with_mask=True, bidir=True)),
    ("+ Masking + Conv + Bidir", dict(with_mask=True, bidir=True, conv=True)),
]
for name, kw in configs:
    m = make_model(**kw)
    m.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    acc = m.evaluate(x[:150], y[:150], verbose=0)[1]
    params = m.count_params()
    print(f"{name:28s} acc={acc:.3f} params={params}")
# النتيجة المتوقعة: الدقّة تتحسّن تدريجيًّا، وعدد المعاملات يتصاعد
# مع Bidirectional (×2) ويقلّ مع Conv (تقليص الطول).
