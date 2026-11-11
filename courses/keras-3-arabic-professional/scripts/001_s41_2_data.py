# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.2 البيانات
# المصدر  : src/chapters/ch41_project_vision.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: صور حقيقيّة)
n = 4000
x = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. تقسيم زمنيّ (في الإنتاج: بحسب تاريخ التصوير)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

# ─── 3. موازنة الأصناف
weights = {0: 1.0, 1: float(len(y_train) / (2 * y_train.sum()))}
# النتيجة المتوقعة: وزن الصنف النادر ≈ 2.5.
