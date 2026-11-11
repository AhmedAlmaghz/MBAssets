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
