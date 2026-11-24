#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — مشاريع Keras عملية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 25 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-25-ch25 (lesson 25)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-25-ch25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — مشاريع Keras عملية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 25 — Keras

Book slug : pytorch-keras-deep-learning
Chapter   : ch-25-ch25 (lesson 25)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

# 1) البيانات: قاعدةFashion-MNIST المدمجة في Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2) التحضير: توحيد الأبعاد + التطبيع
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print("train:", x_train.shape, "test:", x_test.shape)
# الناتج: train: (60000, 784) test: (10000, 784)

# 3) النموذج: MLP بسيط أولاً
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10),                       # logits بلا softmax
])

# 4) الترجمة (compile)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# 5) التدريب مع الإنذار المبكر
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=0,
)

# 6) التقييم النهائي على بيانات الاختبار
results = model.evaluate(x_test, y_test, return_dict=True)
print(f"test loss={results['loss']:.4f}  test acc={results['acc']:.4f}")
# الناتج التقريبي: test loss=0.3201  test acc=0.8846

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) البيانات والتقسيم
data = fetch_california_housing()
X, y = data.data, data.target          # y = السعر بالآلاف من الدولارات
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) التطبيع (على التدريب فقط! التسرّب هنا شائع جداً)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# 3) خطّ الأساس: متوسط التدريب
baseline = np.full_like(yte, ytr.mean())
rmse_base = np.sqrt(np.mean((yte - baseline) ** 2))
print(f"baseline RMSE: {rmse_base:.4f}")     # الناتج: baseline RMSE: 0.7456

# 4) النموذج
model = keras.Sequential([
    keras.layers.Input(shape=(Xtr_s.shape[1],)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1),                  # مخرج واحد: القيمة المتوقّعة
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)

history = model.fit(Xtr_s, ytr, validation_split=0.15, epochs=60,
                    batch_size=256,
                    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
                    verbose=0)

pred = model.predict(Xte_s, verbose=0).ravel()
rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
mae = float(np.mean(np.abs(yte - pred)))
print(f"model RMSE: {rmse:.4f}  MAE: {mae:.4f}")
# الناتج التقريبي: model RMSE: 0.4650  MAE: 0.3521

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# 1) البيانات المدمجة: مراجعات IMDB
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.utils.pad_sequences(x_train, maxlen=200)
x_test = keras.utils.pad_sequences(x_test, maxlen=200)
print("shape:", x_train.shape)    # (25000, 200)

# 2) نموذج: تضمين + GlobalAveragePooling + مصنّف
model = keras.Sequential([
    keras.layers.Input(shape=(200,)),
    keras.layers.Embedding(input_dim=10000, output_dim=64),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

model.fit(x_train, y_train, validation_split=0.2, epochs=8, batch_size=64,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
          verbose=0)

acc = model.evaluate(x_test, y_test, return_dict=True)["acc"]
print(f"test acc: {acc:.4f}")     # الناتج: test acc: 0.8637

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بيانات: طلب يومي على 3 سنوات (محاكاة)
rng = np.random.default_rng(0)
n = 1095
t = np.arange(n)
series = (50 + 10 * np.sin(2 * np.pi * t / 7)
          + 0.05 * t
          + rng.normal(scale=3, size=n))

def make_windows(arr, window=28, horizon=1):
    X, Y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i + window])
        Y.append(arr[i + window:i + window + horizon])
    return np.array(X, dtype="float32"), np.array(Y, dtype="float32")

X, Y = make_windows(series)
split = int(0.8 * len(X))
Xtr, Xte, Ytr, Yte = X[:split], X[split:], Y[:split], Y[split:]

# توحيد على التدريب فقط
mean, std = Xtr.mean(), Xtr.std()
Xtr_n, Xte_n = (Xtr - mean) / std, (Xte - mean) / std

model = keras.Sequential([
    keras.layers.Input(shape=(28, 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
               loss=keras.losses.MeanSquaredError())

model.fit(Xtr_n, Ytr, validation_split=0.1, epochs=30, batch_size=32,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=0)

pred = model.predict(Xte_n, verbose=0).ravel() * std + mean
rmse = float(np.sqrt(np.mean((Yte.ravel() - pred) ** 2)))
naive = float(np.sqrt(np.mean((Yte.ravel() - Xte[:, -1]) ** 2)))   # تنبّؤ ساذج
print(f"LSTM RMSE: {rmse:.2f}   naive RMSE: {naive:.2f}")
# الناتج التقريبي: LSTM RMSE: 3.02   naive RMSE: 3.98

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بيانات جدولية: 100 ألف عميل، 20 خاصية، هدف ثنائي (تعثّر)
rng = np.random.default_rng(7)
n, d = 100_000, 20
X = rng.normal(size=(n, d)).astype("float32")
logit = (X[:, 0] * 1.4 + X[:, 1] * 0.9 - X[:, 2] * 0.6
         + rng.normal(scale=0.4, size=n))
y = (1 / (1 + np.exp(-logit)) > 0.5).astype("float32")

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = keras.Sequential([
    keras.layers.Input(shape=(d,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                        keras.metrics.AUC(name="auc")])

model.fit(Xtr, ytr, validation_split=0.15, epochs=25, batch_size=1024,
          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
          verbose=0)

r = model.evaluate(Xte, yte, return_dict=True)
print(f"acc={r['acc']:.4f}  auc={r['auc']:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مولّد ومميّز بسيطان على Fashion-MNIST (28×28)
G = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(784, activation="sigmoid"),
])

D = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1),                       # logits: حقيقي أم مزيّف؟
])

G.compile(optimizer=keras.optimizers.Adam(2e-4, 0.5), loss=keras.losses.BinaryCrossentropy(from_logits=True))
D.compile(optimizer=keras.optimizers.Adam(2e-4, 0.5), loss=keras.losses.BinaryCrossentropy(from_logits=True))

(x_train, _), _ = keras.datasets.fashion_mnist.load_data()
real = x_train.astype("float32") / 255.0

BATCH = 64
def train_step(noise, real_batch):
    fake = G(noise, training=True)
    d_real = D(real_batch, training=True)
    d_fake = D(fake, training=True)
    d_loss = 0.5 * (D.compute_loss(None, d_real, real_batch)
                   + D.compute_loss(None, d_fake, np.zeros_like(d_fake)))
    g_loss = D.compute_loss(None, D(fake, training=True), np.ones((BATCH, 1)))
    return d_loss, g_loss

for step in range(2000):
    idx = np.random.randint(0, len(real), BATCH)
    noise = np.random.normal(size=(BATCH, 100)).astype("float32")
    d_loss, g_loss = train_step(noise, real[idx])
    if step % 500 == 0:
        print(f"step {step}: D={d_loss:.3f} G={g_loss:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch25.html
# ────────────────────────────────────────────────────────────────────────
# النسخة الخاطئة: التطبيع على كامل البيانات قبل التقسيم
sc_bad = StandardScaler().fit(X)          # سمح لـX_test بالتأثير في المعاملات!
X_bad = sc_bad.transform(X)
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_bad, y, test_size=0.2, random_state=42)

# النسخة الصحيحة
sc_ok = StandardScaler().fit(Xtr)
Xtr_ok, Xte_ok = sc.transform(Xtr), sc.transform(Xte)

# الفرق: الأرقام إحصائياً قريبة، لكن «التسرّب» يعني أن أداء الاختبار
# لم يعد قياساً نزيهاً — وهو خطأ منهجي لا يظهر بالضرورة في الأرقام!

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

# 1) البيانات: قاعدةFashion-MNIST المدمجة في Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2) التحضير: توحيد الأبعاد + التطبيع
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print("train:", x_train.shape, "test:", x_test.shape)
# الناتج: train: (60000, 784) test: (10000, 784)

# 3) النموذج: MLP بسيط أولاً
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10),                       # logits بلا softmax
])

# 4) الترجمة (compile)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# 5) التدريب مع الإنذار المبكر
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=0,
)

# 6) التقييم النهائي على بيانات الاختبار
results = model.evaluate(x_test, y_test, return_dict=True)
print(f"test loss={results['loss']:.4f}  test acc={results['acc']:.4f}")
# الناتج التقريبي: test loss=0.3201  test acc=0.8846

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) البيانات والتقسيم
data = fetch_california_housing()
X, y = data.data, data.target          # y = السعر بالآلاف من الدولارات
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) التطبيع (على التدريب فقط! التسرّب هنا شائع جداً)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# 3) خطّ الأساس: متوسط التدريب
baseline = np.full_like(yte, ytr.mean())
rmse_base = np.sqrt(np.mean((yte - baseline) ** 2))
print(f"baseline RMSE: {rmse_base:.4f}")     # الناتج: baseline RMSE: 0.7456

# 4) النموذج
model = keras.Sequential([
    keras.layers.Input(shape=(Xtr_s.shape[1],)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1),                  # مخرج واحد: القيمة المتوقّعة
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)

history = model.fit(Xtr_s, ytr, validation_split=0.15, epochs=60,
                    batch_size=256,
                    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
                    verbose=0)

pred = model.predict(Xte_s, verbose=0).ravel()
rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
mae = float(np.mean(np.abs(yte - pred)))
print(f"model RMSE: {rmse:.4f}  MAE: {mae:.4f}")
# الناتج التقريبي: model RMSE: 0.4650  MAE: 0.3521

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# 1) البيانات المدمجة: مراجعات IMDB
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.utils.pad_sequences(x_train, maxlen=200)
x_test = keras.utils.pad_sequences(x_test, maxlen=200)
print("shape:", x_train.shape)    # (25000, 200)

# 2) نموذج: تضمين + GlobalAveragePooling + مصنّف
model = keras.Sequential([
    keras.layers.Input(shape=(200,)),
    keras.layers.Embedding(input_dim=10000, output_dim=64),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

model.fit(x_train, y_train, validation_split=0.2, epochs=8, batch_size=64,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
          verbose=0)

acc = model.evaluate(x_test, y_test, return_dict=True)["acc"]
print(f"test acc: {acc:.4f}")     # الناتج: test acc: 0.8637

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بيانات: طلب يومي على 3 سنوات (محاكاة)
rng = np.random.default_rng(0)
n = 1095
t = np.arange(n)
series = (50 + 10 * np.sin(2 * np.pi * t / 7)
          + 0.05 * t
          + rng.normal(scale=3, size=n))

def make_windows(arr, window=28, horizon=1):
    X, Y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i + window])
        Y.append(arr[i + window:i + window + horizon])
    return np.array(X, dtype="float32"), np.array(Y, dtype="float32")

X, Y = make_windows(series)
split = int(0.8 * len(X))
Xtr, Xte, Ytr, Yte = X[:split], X[split:], Y[:split], Y[split:]

# توحيد على التدريب فقط
mean, std = Xtr.mean(), Xtr.std()
Xtr_n, Xte_n = (Xtr - mean) / std, (Xte - mean) / std

model = keras.Sequential([
    keras.layers.Input(shape=(28, 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
               loss=keras.losses.MeanSquaredError())

model.fit(Xtr_n, Ytr, validation_split=0.1, epochs=30, batch_size=32,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=0)

pred = model.predict(Xte_n, verbose=0).ravel() * std + mean
rmse = float(np.sqrt(np.mean((Yte.ravel() - pred) ** 2)))
naive = float(np.sqrt(np.mean((Yte.ravel() - Xte[:, -1]) ** 2)))   # تنبّؤ ساذج
print(f"LSTM RMSE: {rmse:.2f}   naive RMSE: {naive:.2f}")
# الناتج التقريبي: LSTM RMSE: 3.02   naive RMSE: 3.98

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بيانات جدولية: 100 ألف عميل، 20 خاصية، هدف ثنائي (تعثّر)
rng = np.random.default_rng(7)
n, d = 100_000, 20
X = rng.normal(size=(n, d)).astype("float32")
logit = (X[:, 0] * 1.4 + X[:, 1] * 0.9 - X[:, 2] * 0.6
         + rng.normal(scale=0.4, size=n))
y = (1 / (1 + np.exp(-logit)) > 0.5).astype("float32")

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = keras.Sequential([
    keras.layers.Input(shape=(d,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                        keras.metrics.AUC(name="auc")])

model.fit(Xtr, ytr, validation_split=0.15, epochs=25, batch_size=1024,
          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
          verbose=0)

r = model.evaluate(Xte, yte, return_dict=True)
print(f"acc={r['acc']:.4f}  auc={r['auc']:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مولّد ومميّز بسيطان على Fashion-MNIST (28×28)
G = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(784, activation="sigmoid"),
])

D = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1),                       # logits: حقيقي أم مزيّف؟
])

G.compile(optimizer=keras.optimizers.Adam(2e-4, 0.5), loss=keras.losses.BinaryCrossentropy(from_logits=True))
D.compile(optimizer=keras.optimizers.Adam(2e-4, 0.5), loss=keras.losses.BinaryCrossentropy(from_logits=True))

(x_train, _), _ = keras.datasets.fashion_mnist.load_data()
real = x_train.astype("float32") / 255.0

BATCH = 64
def train_step(noise, real_batch):
    fake = G(noise, training=True)
    d_real = D(real_batch, training=True)
    d_fake = D(fake, training=True)
    d_loss = 0.5 * (D.compute_loss(None, d_real, real_batch)
                   + D.compute_loss(None, d_fake, np.zeros_like(d_fake)))
    g_loss = D.compute_loss(None, D(fake, training=True), np.ones((BATCH, 1)))
    return d_loss, g_loss

for step in range(2000):
    idx = np.random.randint(0, len(real), BATCH)
    noise = np.random.normal(size=(BATCH, 100)).astype("float32")
    d_loss, g_loss = train_step(noise, real[idx])
    if step % 500 == 0:
        print(f"step {step}: D={d_loss:.3f} G={g_loss:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-25-ch25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# النسخة الخاطئة: التطبيع على كامل البيانات قبل التقسيم
sc_bad = StandardScaler().fit(X)          # سمح لـX_test بالتأثير في المعاملات!
X_bad = sc_bad.transform(X)
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_bad, y, test_size=0.2, random_state=42)

# النسخة الصحيحة
sc_ok = StandardScaler().fit(Xtr)
Xtr_ok, Xte_ok = sc.transform(Xtr), sc.transform(Xte)

# الفرق: الأرقام إحصائياً قريبة، لكن «التسرّب» يعني أن أداء الاختبار
# لم يعد قياساً نزيهاً — وهو خطأ منهجي لا يظهر بالضرورة في الأرقام!
