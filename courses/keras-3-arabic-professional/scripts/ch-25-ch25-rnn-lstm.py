#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية و LSTM و GRU
Keras 3 in Arabic — Rnn Lstm

Book slug : keras-3-arabic-professional
Chapter   : ch-25-ch25-rnn-lstm (lesson 25)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-25-ch25-rnn-lstm.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية و LSTM و GRU
Keras 3 in Arabic — Rnn Lstm

Book slug : keras-3-arabic-professional
Chapter   : ch-25-ch25-rnn-lstm (lesson 25)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-25-ch25-rnn-lstm.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية و LSTM و GRU
Keras 3 in Arabic — Rnn Lstm

Book slug : keras-3-arabic-professional
Chapter   : ch-25-ch25-rnn-lstm (lesson 25)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch25_rnn_lstm/001_s25_6_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch25_rnn_lstm/002_lstm_validation_versus.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch25_rnn_lstm/003_lstm_gru_versus_data.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الشبكات التكرارية و LSTM و GRU
Keras 3 in Arabic — Rnn Lstm

Book slug : keras-3-arabic-professional
Chapter   : ch-25-ch25-rnn-lstm (lesson 25)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch25_rnn_lstm/001_s25_6_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch25_rnn_lstm/002_lstm_validation_versus.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch25_rnn_lstm/003_lstm_gru_versus_data.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch25_rnn_lstm.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-25-ch25-rnn-lstm.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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
