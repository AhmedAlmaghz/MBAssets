#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قبل الانحدار الخطي: من الخطأ إلى أقل المربعات
Mastering scikit-learn 1.9 — Chapter 27 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-27-ch27 (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-27-ch27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قبل الانحدار الخطي: من الخطأ إلى أقل المربعات
Mastering scikit-learn 1.9 — Chapter 27 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-27-ch27 (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# بيانات انحدار بسيطة
X, y = make_regression(n_samples=200, n_features=5,
                        n_informative=5, noise=10, random_state=42)

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"المعاملات (coef_): {lr.coef_.round(3)}")
print(f"الحد الثابت (intercept_): {lr.intercept_:.4f}")
print(f"MSE على التدريب: {mse:.4f}")
print(f"R² على التدريب: {r2:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=3,
                        noise=5, random_state=42)

# الحل اليدوي عبر SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
beta_svd = Vt.T @ np.diag(1/s) @ U.T @ y

# الحل عبر الصيغة الطبيعية
beta_normal = np.linalg.lstsq(X, y, rcond=None)[0]

print("المعاملات عبر SVD   :", beta_svd.round(4))
print("المعاملات عبر lstsq :", beta_normal.round(4))
print("الفارق الأقصى:", np.abs(beta_svd - beta_normal).max())

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=3,
                        noise=15, random_state=42)

lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

print(f"متوسط المتبقيات: {residuals.mean():.8f}  (≈ 0 دائماً)")
print(f"انحراف المتبقيات: {residuals.std():.4f}")
print(f"أصغر متبقٍّ: {residuals.min():.4f}")
print(f"أكبر متبقٍّ: {residuals.max():.4f}")

# اختبار بسيط: هل المتبقيات غير مرتبطة بالتنبؤات؟
y_hat = lr.predict(X)
correlation = np.corrcoef(y_hat, residuals)[0, 1]
print(f"\nارتباط المتبقيات بـ ŷ: {correlation:.6f}  (≈ 0 في نموذج جيد)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
lr = LinearRegression()
lr.fit(X, y)
y_hat = lr.predict(X)
residuals = y - y_hat

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(y_hat, residuals, alpha=0.5, s=20)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted values (ŷ)")
axes[0].set_ylabel("Residuals (e)")
axes[0].set_title("Residuals vs. Fitted")

axes[1].hist(residuals, bins=25, edgecolor="white")
axes[1].set_xlabel("Residual value")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("residuals_plot.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=300, n_features=10,
                        n_informative=5, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r2_train = lr.score(X_train, y_train)

print(f"{'المقياس':<20} {'التدريب':>10} {'الاختبار':>12}")
print("-" * 44)
print(f"{'R²':<20} {r2_train:>10.4f} {r2:>12.4f}")
print(f"{'RMSE':<20} {'—':>10} {rmse:>12.4f}")
print(f"{'MAE':<20} {'—':>10} {mae:>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=50,
                        n_informative=5, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

r2_train = lr.score(X_train, y_train)
r2_test = lr.score(X_test, y_test)

print(f"R² التدريب:  {r2_train:.4f}")
print(f"R² الاختبار: {r2_test:.4f}")
print(f"فجوة الإفراط: {r2_train - r2_test:+.4f}")
print()
if r2_train - r2_test > 0.1:
    print("← فجوة كبيرة تشير إلى إفراط في الملاءمة")
    print("   الحل: Ridge أو Lasso لتنظيم المعاملات (الفصل التالي)")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part04/ch27.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=3,
                        noise=5, random_state=42)
# إضافة عمود تحيّز
X_b = np.column_stack([np.ones(len(X)), X])

# الحل اليدوي بالصيغة الطبيعية
beta_manual = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# LinearRegression
lr = LinearRegression()
lr.fit(X, y)
beta_sklearn = np.concatenate([[lr.intercept_], lr.coef_])

print("الحل اليدوي:", beta_manual.round(4))
print("LinearRegression:", beta_sklearn.round(4))
print("أقصى فرق:", np.abs(beta_manual - beta_sklearn).max())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# بيانات انحدار بسيطة
X, y = make_regression(n_samples=200, n_features=5,
                        n_informative=5, noise=10, random_state=42)

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"المعاملات (coef_): {lr.coef_.round(3)}")
print(f"الحد الثابت (intercept_): {lr.intercept_:.4f}")
print(f"MSE على التدريب: {mse:.4f}")
print(f"R² على التدريب: {r2:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=3,
                        noise=5, random_state=42)

# الحل اليدوي عبر SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
beta_svd = Vt.T @ np.diag(1/s) @ U.T @ y

# الحل عبر الصيغة الطبيعية
beta_normal = np.linalg.lstsq(X, y, rcond=None)[0]

print("المعاملات عبر SVD   :", beta_svd.round(4))
print("المعاملات عبر lstsq :", beta_normal.round(4))
print("الفارق الأقصى:", np.abs(beta_svd - beta_normal).max())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=3,
                        noise=15, random_state=42)

lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

print(f"متوسط المتبقيات: {residuals.mean():.8f}  (≈ 0 دائماً)")
print(f"انحراف المتبقيات: {residuals.std():.4f}")
print(f"أصغر متبقٍّ: {residuals.min():.4f}")
print(f"أكبر متبقٍّ: {residuals.max():.4f}")

# اختبار بسيط: هل المتبقيات غير مرتبطة بالتنبؤات؟
y_hat = lr.predict(X)
correlation = np.corrcoef(y_hat, residuals)[0, 1]
print(f"\nارتباط المتبقيات بـ ŷ: {correlation:.6f}  (≈ 0 في نموذج جيد)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
lr = LinearRegression()
lr.fit(X, y)
y_hat = lr.predict(X)
residuals = y - y_hat

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(y_hat, residuals, alpha=0.5, s=20)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted values (ŷ)")
axes[0].set_ylabel("Residuals (e)")
axes[0].set_title("Residuals vs. Fitted")

axes[1].hist(residuals, bins=25, edgecolor="white")
axes[1].set_xlabel("Residual value")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("residuals_plot.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=300, n_features=10,
                        n_informative=5, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r2_train = lr.score(X_train, y_train)

print(f"{'المقياس':<20} {'التدريب':>10} {'الاختبار':>12}")
print("-" * 44)
print(f"{'R²':<20} {r2_train:>10.4f} {r2:>12.4f}")
print(f"{'RMSE':<20} {'—':>10} {rmse:>12.4f}")
print(f"{'MAE':<20} {'—':>10} {mae:>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=50,
                        n_informative=5, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

r2_train = lr.score(X_train, y_train)
r2_test = lr.score(X_test, y_test)

print(f"R² التدريب:  {r2_train:.4f}")
print(f"R² الاختبار: {r2_test:.4f}")
print(f"فجوة الإفراط: {r2_train - r2_test:+.4f}")
print()
if r2_train - r2_test > 0.1:
    print("← فجوة كبيرة تشير إلى إفراط في الملاءمة")
    print("   الحل: Ridge أو Lasso لتنظيم المعاملات (الفصل التالي)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-27-ch27.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=3,
                        noise=5, random_state=42)
# إضافة عمود تحيّز
X_b = np.column_stack([np.ones(len(X)), X])

# الحل اليدوي بالصيغة الطبيعية
beta_manual = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# LinearRegression
lr = LinearRegression()
lr.fit(X, y)
beta_sklearn = np.concatenate([[lr.intercept_], lr.coef_])

print("الحل اليدوي:", beta_manual.round(4))
print("LinearRegression:", beta_sklearn.round(4))
print("أقصى فرق:", np.abs(beta_manual - beta_sklearn).max())
