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
