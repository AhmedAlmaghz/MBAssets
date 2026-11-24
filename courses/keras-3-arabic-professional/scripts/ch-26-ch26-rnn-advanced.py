#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية المتقدّمة
Keras 3 in Arabic — Rnn Advanced

Book slug : keras-3-arabic-professional
Chapter   : ch-26-ch26-rnn-advanced (lesson 26)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-26-ch26-rnn-advanced.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية المتقدّمة
Keras 3 in Arabic — Rnn Advanced

Book slug : keras-3-arabic-professional
Chapter   : ch-26-ch26-rnn-advanced (lesson 26)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-26-ch26-rnn-advanced.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية المتقدّمة
Keras 3 in Arabic — Rnn Advanced

Book slug : keras-3-arabic-professional
Chapter   : ch-26-ch26-rnn-advanced (lesson 26)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch26_rnn_advanced/001_s26_2_bidirectional_reading.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch26_rnn_advanced/002_s26_3_conv1d_window_local.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch26_rnn_advanced/003_s26_4_attention_lstm.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch26_rnn_advanced/004_s26_5_masking_contamination_state.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch26_rnn_advanced/005_tools_four_data.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية المتقدّمة
Keras 3 in Arabic — Rnn Advanced

Book slug : keras-3-arabic-professional
Chapter   : ch-26-ch26-rnn-advanced (lesson 26)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch26_rnn_advanced/001_s26_2_bidirectional_reading.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch26_rnn_advanced/002_s26_3_conv1d_window_local.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch26_rnn_advanced/003_s26_4_attention_lstm.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch26_rnn_advanced/004_s26_5_masking_contamination_state.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch26_rnn_advanced/005_tools_four_data.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch26_rnn_advanced.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-26-ch26-rnn-advanced.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
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
