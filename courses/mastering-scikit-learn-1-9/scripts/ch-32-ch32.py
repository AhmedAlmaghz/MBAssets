#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المتسامح وتحمّل الصفوف الشاذة
Mastering scikit-learn 1.9 — Chapter 32 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-32-ch32 (lesson 32)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-32-ch32.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المتسامح وتحمّل الصفوف الشاذة
Mastering scikit-learn 1.9 — Chapter 32 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-32-ch32 (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
n_samples = 100
X = rng.uniform(-5, 5, size=(n_samples, 1))
y = 2.5 * X.ravel() + rng.normal(scale=1.0, size=n_samples)

# إضافة 10% نقاط شاذة عنيفة (تلوث)
outlier_idx = rng.choice(n_samples, size=10, replace=False)
y[outlier_idx] += rng.choice([-30.0, 30.0], size=10)

ols = LinearRegression().fit(X, y)
huber = HuberRegressor(epsilon=1.35).fit(X, y)

print(f"الميل الحقيقي: 2.500")
print(f"ميل OLS (المربعات الصغرى): {ols.coef_[0]:.3f} (تأثر بشدة بالشواذ!)")
print(f"ميل Huber المتسامح:        {huber.coef_[0]:.3f} (مقاوم ومقارب للحقيقة)")
print(f"عدد العينات التي اعتبرها هوبر شاذة: {np.sum(huber.outliers_)}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np

rng = np.random.default_rng(42)
X = np.linspace(-10, 10, 150).reshape(-1, 1)
y = 1.8 * X.ravel() + rng.normal(scale=1.5, size=150)
# تلويث 25 عينة
y[rng.choice(150, 25, replace=False)] += rng.normal(scale=40.0, size=25)

ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=5,
                         residual_threshold=5.0, random_state=42)
ransac.fit(X, y)

inliers = ransac.inlier_mask_
outliers = ~inliers

print(f"إجمالي العينات: {len(X)}")
print(f"العينات السليمة المعتمدة (Inliers): {np.sum(inliers)}")
print(f"العينات الشاذة المكتشفة والمعزولة (Outliers): {np.sum(outliers)}")
print(f"ميل خط RANSAC النقي: {ransac.estimator_.coef_[0]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import TheilSenRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(80, 1))
y = 3.2 * X.ravel() + rng.normal(scale=0.8, size=80)

# تلويث X و y معاً برافعة شاذة قوية
X[0, 0] = 12.0
y[0] = -50.0

theil_sen = TheilSenRegressor(random_state=42).fit(X, y)
print(f"ميل Theil-Sen بعد التلويث برافعة شاذة: {theil_sen.coef_[0]:.3f} (الحقيقي: 3.200)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import QuantileRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(100, 1))
# تباين يتسع تدريجياً (Heteroscedasticity)
y = 2.0 * X.ravel() + rng.normal(scale=0.5 * X.ravel() + 0.1, size=100)

qr_median = QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
qr_upper = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y)

print(f"الميل عند الوسيط (quantile=0.5): {qr_median.coef_[0]:.3f}")
print(f"الميل عند المئين 90% (quantile=0.9): {qr_upper.coef_[0]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
X_clean = rng.uniform(-5, 5, size=(200, 1))
y_clean = 3.0 * X_clean.ravel() + rng.normal(scale=1.0, size=200)

# نسخة ملوثة للتدريب
X_train = X_clean.copy()
y_train = y_clean.copy()
bad_idx = rng.choice(200, 30, replace=False)
y_train[bad_idx] += rng.uniform(40, 80, size=30)

# بيانات اختبار نقية تماماً
X_test = rng.uniform(-5, 5, size=(100, 1))
y_test = 3.0 * X_test.ravel() + rng.normal(scale=1.0, size=100)

ols = LinearRegression().fit(X_train, y_train)
huber = HuberRegressor().fit(X_train, y_train)
ransac = RANSACRegressor(random_state=42).fit(X_train, y_train)

print(f"MAE لـ OLS على الاختبار النقي:    {mean_absolute_error(y_test, ols.predict(X_test)):.3f}")
print(f"MAE لـ Huber على الاختبار النقي:  {mean_absolute_error(y_test, huber.predict(X_test)):.3f}")
print(f"MAE لـ RANSAC على الاختبار النقي: {mean_absolute_error(y_test, ransac.predict(X_test)):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch32.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
# استخدام 300 عينة وميزة متوسط الدخل فقط للإيضاح السريع
X = np.asarray(data.data[:300, [0]])
y = np.asarray(data.target[:300])

q_low = QuantileRegressor(quantile=0.1, alpha=0.0).fit(X, y)
q_high = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y)

y_low = q_low.predict(X)
y_high = q_high.predict(X)

inside = np.mean((y >= y_low) & (y <= y_high))
print(f"نسبة البيانات الواقعة داخل شريط المئينات [10%, 90%]: {inside:.1%}")
print("النسبة النظرية المتوقعة: 80.0%")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
n_samples = 100
X = rng.uniform(-5, 5, size=(n_samples, 1))
y = 2.5 * X.ravel() + rng.normal(scale=1.0, size=n_samples)

# إضافة 10% نقاط شاذة عنيفة (تلوث)
outlier_idx = rng.choice(n_samples, size=10, replace=False)
y[outlier_idx] += rng.choice([-30.0, 30.0], size=10)

ols = LinearRegression().fit(X, y)
huber = HuberRegressor(epsilon=1.35).fit(X, y)

print(f"الميل الحقيقي: 2.500")
print(f"ميل OLS (المربعات الصغرى): {ols.coef_[0]:.3f} (تأثر بشدة بالشواذ!)")
print(f"ميل Huber المتسامح:        {huber.coef_[0]:.3f} (مقاوم ومقارب للحقيقة)")
print(f"عدد العينات التي اعتبرها هوبر شاذة: {np.sum(huber.outliers_)}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np

rng = np.random.default_rng(42)
X = np.linspace(-10, 10, 150).reshape(-1, 1)
y = 1.8 * X.ravel() + rng.normal(scale=1.5, size=150)
# تلويث 25 عينة
y[rng.choice(150, 25, replace=False)] += rng.normal(scale=40.0, size=25)

ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=5,
                         residual_threshold=5.0, random_state=42)
ransac.fit(X, y)

inliers = ransac.inlier_mask_
outliers = ~inliers

print(f"إجمالي العينات: {len(X)}")
print(f"العينات السليمة المعتمدة (Inliers): {np.sum(inliers)}")
print(f"العينات الشاذة المكتشفة والمعزولة (Outliers): {np.sum(outliers)}")
print(f"ميل خط RANSAC النقي: {ransac.estimator_.coef_[0]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import TheilSenRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(80, 1))
y = 3.2 * X.ravel() + rng.normal(scale=0.8, size=80)

# تلويث X و y معاً برافعة شاذة قوية
X[0, 0] = 12.0
y[0] = -50.0

theil_sen = TheilSenRegressor(random_state=42).fit(X, y)
print(f"ميل Theil-Sen بعد التلويث برافعة شاذة: {theil_sen.coef_[0]:.3f} (الحقيقي: 3.200)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import QuantileRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(100, 1))
# تباين يتسع تدريجياً (Heteroscedasticity)
y = 2.0 * X.ravel() + rng.normal(scale=0.5 * X.ravel() + 0.1, size=100)

qr_median = QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
qr_upper = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y)

print(f"الميل عند الوسيط (quantile=0.5): {qr_median.coef_[0]:.3f}")
print(f"الميل عند المئين 90% (quantile=0.9): {qr_upper.coef_[0]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
X_clean = rng.uniform(-5, 5, size=(200, 1))
y_clean = 3.0 * X_clean.ravel() + rng.normal(scale=1.0, size=200)

# نسخة ملوثة للتدريب
X_train = X_clean.copy()
y_train = y_clean.copy()
bad_idx = rng.choice(200, 30, replace=False)
y_train[bad_idx] += rng.uniform(40, 80, size=30)

# بيانات اختبار نقية تماماً
X_test = rng.uniform(-5, 5, size=(100, 1))
y_test = 3.0 * X_test.ravel() + rng.normal(scale=1.0, size=100)

ols = LinearRegression().fit(X_train, y_train)
huber = HuberRegressor().fit(X_train, y_train)
ransac = RANSACRegressor(random_state=42).fit(X_train, y_train)

print(f"MAE لـ OLS على الاختبار النقي:    {mean_absolute_error(y_test, ols.predict(X_test)):.3f}")
print(f"MAE لـ Huber على الاختبار النقي:  {mean_absolute_error(y_test, huber.predict(X_test)):.3f}")
print(f"MAE لـ RANSAC على الاختبار النقي: {mean_absolute_error(y_test, ransac.predict(X_test)):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-32-ch32.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
# استخدام 300 عينة وميزة متوسط الدخل فقط للإيضاح السريع
X = np.asarray(data.data[:300, [0]])
y = np.asarray(data.target[:300])

q_low = QuantileRegressor(quantile=0.1, alpha=0.0).fit(X, y)
q_high = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y)

y_low = q_low.predict(X)
y_high = q_high.predict(X)

inside = np.mean((y >= y_low) & (y <= y_high))
print(f"نسبة البيانات الواقعة داخل شريط المئينات [10%, 90%]: {inside:.1%}")
print("النسبة النظرية المتوقعة: 80.0%")
