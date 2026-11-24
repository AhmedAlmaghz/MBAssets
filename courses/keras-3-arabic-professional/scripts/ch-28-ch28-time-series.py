#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — السلاسل الزمنية
Keras 3 in Arabic — Time Series

Book slug : keras-3-arabic-professional
Chapter   : ch-28-ch28-time-series (lesson 28)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-28-ch28-time-series.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — السلاسل الزمنية
Keras 3 in Arabic — Time Series

Book slug : keras-3-arabic-professional
Chapter   : ch-28-ch28-time-series (lesson 28)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-28-ch28-time-series.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — السلاسل الزمنية
Keras 3 in Arabic — Time Series

Book slug : keras-3-arabic-professional
Chapter   : ch-28-ch28-time-series (lesson 28)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch28_time_series/001_s28_4_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : 28.4 التنفيذ في Keras
# المصدر  : src/chapters/ch28_time_series.md:56
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch28_time_series/002_comparison_pipeline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : المقارنة الكاملة على خطّ أساسين
# المصدر  : src/chapters/ch28_time_series.md:116
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch28_time_series/003_horizon_one_step_versus_ten.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : تأثير الأفق: خطوة واحدة مقابل عشر
# المصدر  : src/chapters/ch28_time_series.md:191
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : 28.4 التنفيذ في Keras
# المصدر  : src/chapters/ch28_time_series.md:56
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : المقارنة الكاملة على خطّ أساسين
# المصدر  : src/chapters/ch28_time_series.md:116
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : تأثير الأفق: خطوة واحدة مقابل عشر
# المصدر  : src/chapters/ch28_time_series.md:191
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — السلاسل الزمنية
Keras 3 in Arabic — Time Series

Book slug : keras-3-arabic-professional
Chapter   : ch-28-ch28-time-series (lesson 28)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch28_time_series/001_s28_4_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : 28.4 التنفيذ في Keras
# المصدر  : src/chapters/ch28_time_series.md:56
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch28_time_series/002_comparison_pipeline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : المقارنة الكاملة على خطّ أساسين
# المصدر  : src/chapters/ch28_time_series.md:116
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch28_time_series/003_horizon_one_step_versus_ten.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : تأثير الأفق: خطوة واحدة مقابل عشر
# المصدر  : src/chapters/ch28_time_series.md:191
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch28_time_series.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : 28.4 التنفيذ في Keras
# المصدر  : src/chapters/ch28_time_series.md:56
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : المقارنة الكاملة على خطّ أساسين
# المصدر  : src/chapters/ch28_time_series.md:116
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 28: السلاسل الزمنية
# القسم   : تأثير الأفق: خطوة واحدة مقابل عشر
# المصدر  : src/chapters/ch28_time_series.md:191
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة زمنيّة مُصنّعة: 1000 خطوة (نمط دوريّ + ضجيج)
T = 1000
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0          # دوريّ
          + np.linspace(0, 5, T)                      # اتجاه صاعد
          + np.random.randn(T)).astype("float32")

# ─── 2. تبويع: نافذة 30 خطوة، وتنبّؤ بالخطوة التالية
WINDOW = 30
def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 3. تقسيم زمنيّ (70/10/20)
train_end = int(0.7 * T)
val_end = int(0.8 * T)
x_train, y_train = make_windows(series[:train_end], WINDOW)
x_val, y_val = make_windows(series[train_end - WINDOW:val_end], WINDOW)
x_test, y_test = make_windows(series[val_end - WINDOW:], WINDOW)

# ─── 4. تطبيع بحسب التدريب فقط (لا تسريب)
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std

# ─── 5. النموذج: LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
out = keras.layers.Dense(1)(x)                       # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=32, verbose=0)

# ─── 6. خطّ الأساس: التكرار السابق
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"naive MAE = {naive_mae:.3f}, model MAE = {model_mae:.3f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# ─── 1. ثلاث تنبّؤات: naive، متوسّط النافذة، LSTM
naive = x_test[:, -1, 0]                    # آخر قيمة
window_avg = x_test.mean(axis=1)[:, 0]      # متوسّط النافذة
lstm_pred = np.array(model.predict(x_test, verbose=0))[:, 0]

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

# ─── 2. الجدول
rows = [("naive (آخر قيمة)", mae(naive, y_test)),
        ("moving average", mae(window_avg, y_test)),
        ("LSTM", mae(lstm_pred, y_test))]
for name, v in rows:
    print(f"{name:22s} MAE = {v:.4f}")
best = min(rows, key=lambda r: r[1])
print(f"الأفضل: {best[0]}")
# النتيجة المتوقعة: LSTM أفضل، وmoving average قريب من naive.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-28-ch28-time-series.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مُصنّعة
T = 600
t = np.arange(T)
series = (np.sin(2 * np.pi * t / 50) * 10.0
          + np.random.randn(T) * 2.0).astype("float32")

def make_windows(arr, window, horizon):
    """نوافذ تتنبّأ بالخطوة t + horizon."""
    xs, ys = [], []
    for i in range(len(arr) - window - horizon + 1):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window + horizon - 1])
    return np.array(xs)[..., None], np.array(ys)

def train_eval(horizon):
    """تدريب وتقييم بأفق محدّد."""
    x, y = make_windows(series[:400], 30, horizon)
    m = keras.Sequential([keras.Input(shape=(30, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    m.fit(x, y, epochs=5, batch_size=32, verbose=0)
    x_t, y_t = make_windows(series[400:], 30, horizon)
    return float(m.evaluate(x_t, y_t, verbose=0)[1])

for h in (1, 5, 10):
    mae = train_eval(h)
    print(f"horizon {h}: MAE = {mae:.4f}")
# النتيجة المتوقعة: MAE يزداد بمرور الأفق — أوسع أُسّيًّا لا خطّيًّا.
