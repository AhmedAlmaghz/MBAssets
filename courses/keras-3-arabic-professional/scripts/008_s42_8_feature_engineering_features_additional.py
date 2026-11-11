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
