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
