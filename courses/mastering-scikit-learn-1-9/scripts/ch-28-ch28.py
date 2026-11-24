#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تشخيص النموذج الخطي
Mastering scikit-learn 1.9 — Chapter 28 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-28-ch28 (lesson 28)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-28-ch28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تشخيص النموذج الخطي
Mastering scikit-learn 1.9 — Chapter 28 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-28-ch28 (lesson 28)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# حساب VIF يدوياً
n_features = X.shape[1]
vif_values = []
for j in range(n_features):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_j = 1 / (1 - r2_j) if r2_j < 1 else float("inf")
    vif_values.append(vif_j)

print(f"{'الميزة':<15} {'VIF':>8}  {'التقييم':>12}")
print("-" * 40)
for name, vif in sorted(zip(feature_names, vif_values), key=lambda t: -t[1]):
    if vif > 10:
        evaluation = "⚠️ مرتفع جداً"
    elif vif > 5:
        evaluation = "⚡ مرتفع"
    else:
        evaluation = "✓ مقبول"
    print(f"{name:<15} {vif:>8.2f}  {evaluation}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from scipy import stats

X, y = make_regression(n_samples=200, n_features=5,
                        noise=15, random_state=42)

lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

# اختبار Shapiro-Wilk لطبيعية المتبقيات
stat, p_value = stats.shapiro(residuals)
print(f"اختبار Shapiro-Wilk:")
print(f"  الإحصاء W = {stat:.4f}")
print(f"  القيمة p  = {p_value:.4f}")
if p_value > 0.05:
    print("  → لا دليل كافٍ على انحراف عن الطبيعية (p > 0.05)")
else:
    print("  → المتبقيات تنحرف عن الطبيعية بشكل دال إحصائياً")

# إحصاءات ربيعية للمقارنة
q25, q75 = np.percentile(residuals, [25, 75])
print(f"\nالانحراف المعياري: {residuals.std():.4f}")
print(f"IQR: {q75 - q25:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from scipy import stats

X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# QQ Plot
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[0].scatter(osm, osr, s=15, alpha=0.6)
axes[0].plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept],
             color="red", linestyle="--")
axes[0].set_xlabel("Theoretical Quantiles")
axes[0].set_ylabel("Sample Quantiles")
axes[0].set_title("Q-Q Plot of Residuals")

# Histogram
axes[1].hist(residuals, bins=25, edgecolor="white", density=True)
x = np.linspace(residuals.min(), residuals.max(), 200)
axes[1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
             color="red", linewidth=2)
axes[1].set_title("Residual Distribution vs Normal")
axes[1].set_xlabel("Residual")

plt.tight_layout()
plt.savefig("qq_plot.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X, y = make_regression(n_samples=100, n_features=2,
                        noise=10, random_state=42)

# أضف نقطتي أثر عاليتين مصطنعتين
X[0] = [10, 10]
y[0] = -200  # قيمة مختلفة جداً

lr = LinearRegression()
lr.fit(X, y)

# حساب Cook's Distance تقريبياً
n, p = X.shape
y_hat = lr.predict(X)
residuals = y - y_hat
mse = (residuals**2).mean()
hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
leverage = np.diag(hat_matrix)

# Cook's Distance باستخدام الصيغة المبسّطة
cooks = (residuals**2 * leverage) / (p * mse * (1 - leverage + 1e-8)**2)
threshold = 4 / n

top5 = np.argsort(cooks)[::-1][:5]
print(f"عتبة Cook's Distance (4/n = {threshold:.4f}):")
print(f"\nأعلى 5 نقاط أثراً:")
for i in top5:
    status = "⚠️ أثر عالٍ" if cooks[i] > threshold else "✓ طبيعية"
    print(f"  العينة #{i:3d}: D = {cooks[i]:.4f}  {status}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)

r2_train = lr.score(X_train_sc, y_train)
r2_test = lr.score(X_test_sc, y_test)
residuals = y_test - lr.predict(X_test_sc)

print(f"R² التدريب: {r2_train:.4f}")
print(f"R² الاختبار: {r2_test:.4f}")
print(f"متوسط المتبقيات: {residuals.mean():.4f}")
print(f"انحراف المتبقيات: {residuals.std():.4f}")

# هل هناك نمط في المتبقيات الكبيرة؟
large_residuals = np.abs(residuals) > 2 * residuals.std()
print(f"\nنسبة المتبقيات الكبيرة (|e| > 2σ): {large_residuals.mean():.2%}")
print("(المتوقع في التوزيع الطبيعي: ~5%)")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# حساب VIF
vif_dict = {}
for j in range(X.shape[1]):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_dict[feature_names[j]] = 1 / (1 - r2_j) if r2_j < 1 else float("inf")

print("VIF لكل ميزة:")
for name, vif in sorted(vif_dict.items(), key=lambda t: -t[1]):
    print(f"  {name:<12}: {vif:.2f}")

# احذف الميزة الأعلى VIF
worst = max(vif_dict, key=vif_dict.get)
idx_worst = feature_names.index(worst)
X_reduced = np.delete(X, idx_worst, axis=1)

lr = LinearRegression()
cv_all = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()
cv_red = cross_val_score(lr, X_reduced, y, cv=5, scoring="r2").mean()
print(f"\nR² CV (كل الميزات): {cv_all:.4f}")
print(f"R² CV (بدون {worst}): {cv_red:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part04/ch28.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X, y = load_diabetes(return_X_y=True)
feature_names = list(load_diabetes().feature_names)

vif_values = []
for j in range(X.shape[1]):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_values.append(1 / (1 - r2_j) if r2_j < 1 else float("inf"))

safe_idx = [i for i, vif in enumerate(vif_values) if vif < 5]
print(f"ميزات VIF < 5: {[feature_names[i] for i in safe_idx]}")

lr = LinearRegression()
r2_all = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()
r2_sel = cross_val_score(lr, X[:, safe_idx], y, cv=5, scoring="r2").mean()
print(f"R² (كل الميزات): {r2_all:.4f}")
print(f"R² (VIF < 5):    {r2_sel:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# حساب VIF يدوياً
n_features = X.shape[1]
vif_values = []
for j in range(n_features):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_j = 1 / (1 - r2_j) if r2_j < 1 else float("inf")
    vif_values.append(vif_j)

print(f"{'الميزة':<15} {'VIF':>8}  {'التقييم':>12}")
print("-" * 40)
for name, vif in sorted(zip(feature_names, vif_values), key=lambda t: -t[1]):
    if vif > 10:
        evaluation = "⚠️ مرتفع جداً"
    elif vif > 5:
        evaluation = "⚡ مرتفع"
    else:
        evaluation = "✓ مقبول"
    print(f"{name:<15} {vif:>8.2f}  {evaluation}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from scipy import stats

X, y = make_regression(n_samples=200, n_features=5,
                        noise=15, random_state=42)

lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

# اختبار Shapiro-Wilk لطبيعية المتبقيات
stat, p_value = stats.shapiro(residuals)
print(f"اختبار Shapiro-Wilk:")
print(f"  الإحصاء W = {stat:.4f}")
print(f"  القيمة p  = {p_value:.4f}")
if p_value > 0.05:
    print("  → لا دليل كافٍ على انحراف عن الطبيعية (p > 0.05)")
else:
    print("  → المتبقيات تنحرف عن الطبيعية بشكل دال إحصائياً")

# إحصاءات ربيعية للمقارنة
q25, q75 = np.percentile(residuals, [25, 75])
print(f"\nالانحراف المعياري: {residuals.std():.4f}")
print(f"IQR: {q75 - q25:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from scipy import stats

X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
lr = LinearRegression()
lr.fit(X, y)
residuals = y - lr.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# QQ Plot
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[0].scatter(osm, osr, s=15, alpha=0.6)
axes[0].plot([osm[0], osm[-1]], [slope*osm[0]+intercept, slope*osm[-1]+intercept],
             color="red", linestyle="--")
axes[0].set_xlabel("Theoretical Quantiles")
axes[0].set_ylabel("Sample Quantiles")
axes[0].set_title("Q-Q Plot of Residuals")

# Histogram
axes[1].hist(residuals, bins=25, edgecolor="white", density=True)
x = np.linspace(residuals.min(), residuals.max(), 200)
axes[1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
             color="red", linewidth=2)
axes[1].set_title("Residual Distribution vs Normal")
axes[1].set_xlabel("Residual")

plt.tight_layout()
plt.savefig("qq_plot.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X, y = make_regression(n_samples=100, n_features=2,
                        noise=10, random_state=42)

# أضف نقطتي أثر عاليتين مصطنعتين
X[0] = [10, 10]
y[0] = -200  # قيمة مختلفة جداً

lr = LinearRegression()
lr.fit(X, y)

# حساب Cook's Distance تقريبياً
n, p = X.shape
y_hat = lr.predict(X)
residuals = y - y_hat
mse = (residuals**2).mean()
hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
leverage = np.diag(hat_matrix)

# Cook's Distance باستخدام الصيغة المبسّطة
cooks = (residuals**2 * leverage) / (p * mse * (1 - leverage + 1e-8)**2)
threshold = 4 / n

top5 = np.argsort(cooks)[::-1][:5]
print(f"عتبة Cook's Distance (4/n = {threshold:.4f}):")
print(f"\nأعلى 5 نقاط أثراً:")
for i in top5:
    status = "⚠️ أثر عالٍ" if cooks[i] > threshold else "✓ طبيعية"
    print(f"  العينة #{i:3d}: D = {cooks[i]:.4f}  {status}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)

r2_train = lr.score(X_train_sc, y_train)
r2_test = lr.score(X_test_sc, y_test)
residuals = y_test - lr.predict(X_test_sc)

print(f"R² التدريب: {r2_train:.4f}")
print(f"R² الاختبار: {r2_test:.4f}")
print(f"متوسط المتبقيات: {residuals.mean():.4f}")
print(f"انحراف المتبقيات: {residuals.std():.4f}")

# هل هناك نمط في المتبقيات الكبيرة؟
large_residuals = np.abs(residuals) > 2 * residuals.std()
print(f"\nنسبة المتبقيات الكبيرة (|e| > 2σ): {large_residuals.mean():.2%}")
print("(المتوقع في التوزيع الطبيعي: ~5%)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# حساب VIF
vif_dict = {}
for j in range(X.shape[1]):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_dict[feature_names[j]] = 1 / (1 - r2_j) if r2_j < 1 else float("inf")

print("VIF لكل ميزة:")
for name, vif in sorted(vif_dict.items(), key=lambda t: -t[1]):
    print(f"  {name:<12}: {vif:.2f}")

# احذف الميزة الأعلى VIF
worst = max(vif_dict, key=vif_dict.get)
idx_worst = feature_names.index(worst)
X_reduced = np.delete(X, idx_worst, axis=1)

lr = LinearRegression()
cv_all = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()
cv_red = cross_val_score(lr, X_reduced, y, cv=5, scoring="r2").mean()
print(f"\nR² CV (كل الميزات): {cv_all:.4f}")
print(f"R² CV (بدون {worst}): {cv_red:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-28-ch28.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X, y = load_diabetes(return_X_y=True)
feature_names = list(load_diabetes().feature_names)

vif_values = []
for j in range(X.shape[1]):
    X_others = np.delete(X, j, axis=1)
    lr_vif = LinearRegression()
    lr_vif.fit(X_others, X[:, j])
    r2_j = lr_vif.score(X_others, X[:, j])
    vif_values.append(1 / (1 - r2_j) if r2_j < 1 else float("inf"))

safe_idx = [i for i, vif in enumerate(vif_values) if vif < 5]
print(f"ميزات VIF < 5: {[feature_names[i] for i in safe_idx]}")

lr = LinearRegression()
r2_all = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()
r2_sel = cross_val_score(lr, X[:, safe_idx], y, cv=5, scoring="r2").mean()
print(f"R² (كل الميزات): {r2_all:.4f}")
print(f"R² (VIF < 5):    {r2_sel:.4f}")
