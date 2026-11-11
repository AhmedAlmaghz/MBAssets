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
