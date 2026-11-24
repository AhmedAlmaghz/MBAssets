#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع التنبّؤ بالسلاسل الزمنيّة
Keras 3 in Arabic — Project Timeseries

Book slug : keras-3-arabic-professional
Chapter   : ch-42-ch42-project-timeseries (lesson 42)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-42-ch42-project-timeseries.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع التنبّؤ بالسلاسل الزمنيّة
Keras 3 in Arabic — Project Timeseries

Book slug : keras-3-arabic-professional
Chapter   : ch-42-ch42-project-timeseries (lesson 42)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-42-ch42-project-timeseries.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع التنبّؤ بالسلاسل الزمنيّة
Keras 3 in Arabic — Project Timeseries

Book slug : keras-3-arabic-professional
Chapter   : ch-42-ch42-project-timeseries (lesson 42)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch42_project_timeseries/001_s42_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.2 البيانات
# المصدر  : src/chapters/ch42_project_timeseries.md:31
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch42_project_timeseries/002_s42_3_windowing_temporal.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.3 التبويع والتقسيم الزمنيّ
# المصدر  : src/chapters/ch42_project_timeseries.md:54
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch42_project_timeseries/003_s42_4_pipeline_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.4 خطّ الأساس
# المصدر  : src/chapters/ch42_project_timeseries.md:92
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch42_project_timeseries/004_s42_5_lstm_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.5 نموذج LSTM
# المصدر  : src/chapters/ch42_project_timeseries.md:107
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch42_project_timeseries/005_s42_6_walk_forward_test_walk_forward.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.6 الاختبار العكسيّ (Walk-Forward)
# المصدر  : src/chapters/ch42_project_timeseries.md:132
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch42_project_timeseries/006_s42_7_interval_forecast.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.7 فاصل التنبّؤ
# المصدر  : src/chapters/ch42_project_timeseries.md:174
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch42_project_timeseries/007_coverage_interval.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : تحقّق تغطية الفاصل
# المصدر  : src/chapters/ch42_project_timeseries.md:191
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch42_project_timeseries/008_s42_8_feature_engineering_features_additional.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.8 الميزات الإضافيّة (Feature Engineering)
# المصدر  : src/chapters/ch42_project_timeseries.md:224
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch42_project_timeseries/009_s42_9_drift_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.9 الانحراف وإعادة التدريب
# المصدر  : src/chapters/ch42_project_timeseries.md:249
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.2 البيانات
# المصدر  : src/chapters/ch42_project_timeseries.md:31
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.3 التبويع والتقسيم الزمنيّ
# المصدر  : src/chapters/ch42_project_timeseries.md:54
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.4 خطّ الأساس
# المصدر  : src/chapters/ch42_project_timeseries.md:92
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.5 نموذج LSTM
# المصدر  : src/chapters/ch42_project_timeseries.md:107
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.6 الاختبار العكسيّ (Walk-Forward)
# المصدر  : src/chapters/ch42_project_timeseries.md:132
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.7 فاصل التنبّؤ
# المصدر  : src/chapters/ch42_project_timeseries.md:174
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : تحقّق تغطية الفاصل
# المصدر  : src/chapters/ch42_project_timeseries.md:191
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.8 الميزات الإضافيّة (Feature Engineering)
# المصدر  : src/chapters/ch42_project_timeseries.md:224
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.9 الانحراف وإعادة التدريب
# المصدر  : src/chapters/ch42_project_timeseries.md:249
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع التنبّؤ بالسلاسل الزمنيّة
Keras 3 in Arabic — Project Timeseries

Book slug : keras-3-arabic-professional
Chapter   : ch-42-ch42-project-timeseries (lesson 42)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch42_project_timeseries/001_s42_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.2 البيانات
# المصدر  : src/chapters/ch42_project_timeseries.md:31
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch42_project_timeseries/002_s42_3_windowing_temporal.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.3 التبويع والتقسيم الزمنيّ
# المصدر  : src/chapters/ch42_project_timeseries.md:54
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch42_project_timeseries/003_s42_4_pipeline_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.4 خطّ الأساس
# المصدر  : src/chapters/ch42_project_timeseries.md:92
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch42_project_timeseries/004_s42_5_lstm_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.5 نموذج LSTM
# المصدر  : src/chapters/ch42_project_timeseries.md:107
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch42_project_timeseries/005_s42_6_walk_forward_test_walk_forward.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.6 الاختبار العكسيّ (Walk-Forward)
# المصدر  : src/chapters/ch42_project_timeseries.md:132
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch42_project_timeseries/006_s42_7_interval_forecast.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.7 فاصل التنبّؤ
# المصدر  : src/chapters/ch42_project_timeseries.md:174
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch42_project_timeseries/007_coverage_interval.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : تحقّق تغطية الفاصل
# المصدر  : src/chapters/ch42_project_timeseries.md:191
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch42_project_timeseries/008_s42_8_feature_engineering_features_additional.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.8 الميزات الإضافيّة (Feature Engineering)
# المصدر  : src/chapters/ch42_project_timeseries.md:224
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch42_project_timeseries/009_s42_9_drift_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.9 الانحراف وإعادة التدريب
# المصدر  : src/chapters/ch42_project_timeseries.md:249
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch42_project_timeseries.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.2 البيانات
# المصدر  : src/chapters/ch42_project_timeseries.md:31
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.3 التبويع والتقسيم الزمنيّ
# المصدر  : src/chapters/ch42_project_timeseries.md:54
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.4 خطّ الأساس
# المصدر  : src/chapters/ch42_project_timeseries.md:92
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.5 نموذج LSTM
# المصدر  : src/chapters/ch42_project_timeseries.md:107
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.6 الاختبار العكسيّ (Walk-Forward)
# المصدر  : src/chapters/ch42_project_timeseries.md:132
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.7 فاصل التنبّؤ
# المصدر  : src/chapters/ch42_project_timeseries.md:174
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : تحقّق تغطية الفاصل
# المصدر  : src/chapters/ch42_project_timeseries.md:191
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.8 الميزات الإضافيّة (Feature Engineering)
# المصدر  : src/chapters/ch42_project_timeseries.md:224
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.9 الانحراف وإعادة التدريب
# المصدر  : src/chapters/ch42_project_timeseries.md:249
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. سلسلة مبيعات مُصنّعة (في الإنتاج: بيانات حقيقيّة)
T = 260
t = np.arange(T)
sales = (100.0
         + 20.0 * np.sin(2 * np.pi * t / 52)     # دوريّ سنويّ
         + np.linspace(0, 15, T)                  # اتجاه صاعد
         + np.random.randn(T) * 5.0).astype("float32")

# ─── 2. تنظيف: إزالة قيم شاذّة (في الإنتاج: من مصدر بيانات)
q1, q3 = np.percentile(sales, [25, 75])
iqr = q3 - q1
sales = np.clip(sales, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# النتيجة المتوقعة: سلسلة بلا قيم شاذّة.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
WINDOW = 26

def make_windows(arr, window):
    """يحوّل السلسلة إلى نوافذ (X, y) بترتيب زمنيّ."""
    xs, ys = [], []
    for i in range(len(arr) - window):
        xs.append(arr[i:i + window])
        ys.append(arr[i + window])
    return np.array(xs)[..., None], np.array(ys)

# ─── 1. تقسيم زمنيّ (70/10/20)
n_train = int(0.7 * T)
n_val = int(0.1 * T)
train = sales[:n_train]
val = sales[n_train - WINDOW:n_train + n_val]
test = sales[n_train + n_val - WINDOW:]

# ─── 2. التبويع على كلّ فترة (بعد التقسيم)
x_train, y_train = make_windows(train, WINDOW)
x_val, y_val = make_windows(val, WINDOW)
x_test, y_test = make_windows(test, WINDOW)

# ─── 3. تطبيع بحسب التدريب فقط
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean) / std
y_val = (y_val - mean) / std
y_test = (y_test - mean) / std
# النتيجة المتوقعة: نوافذ مُطبَّعة بلا تسريب.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حساب فاصل التنبّؤ على مجموعة تحقّق
val_pred = model.predict(x_val, verbose=0)[:, 0]
residuals = y_val - val_pred
sigma = np.std(residuals)
# النتيجة المتوقعة: انحراف معياريّ للأخطاء.

# ─── 2. فاصل 90% (بافتراض توزيع غاوسيّ)
lower = val_pred - 1.64 * sigma
upper = val_pred + 1.64 * sigma
# النتيجة المتوقعة: 90% من القيم الحقيقيّة داخل الفاصل.

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def add_features(series):
    """يُضيف ميزات زمنيّة إلى السلسلة."""
    n = len(series)
    t = np.arange(n)
    dow = (t % 52) % 7                       # يوم أسبوعيّ مبسّط
    month = (t % 52) // 4.33                 # شهر مبسّط
    lag1 = np.roll(series, 1)
    avg4 = np.convolve(series, np.ones(4) / 4, mode="same")
    feats = np.stack([series, dow, month, lag1, avg4], axis=-1)
    return feats.astype("float32")

feats = add_features(np.arange(100, dtype="float32"))
print(feats.shape)   # (100, 5)
# النتيجة المتوقعة: 5 ميزات لكلّ خطوة زمنيّة.

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-42-ch42-project-timeseries.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.
