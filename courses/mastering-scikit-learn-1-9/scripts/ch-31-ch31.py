#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المرن والمتعدد المهام
Mastering scikit-learn 1.9 — Chapter 31 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-31-ch31 (lesson 31)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-31-ch31.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المرن والمتعدد المهام
Mastering scikit-learn 1.9 — Chapter 31 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-31-ch31 (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.default_rng(42)
n_samples, n_features, n_tasks = 120, 30, 4

# مصفوفة بيانات مشتركة
X = rng.normal(size=(n_samples, n_features))

# 4 ميزات فقط هي الحقيقية المؤثرة في كل الأهداف
true_W = np.zeros((n_features, n_tasks))
true_W[[2, 7, 14, 21], :] = rng.uniform(2.0, 5.0, size=(4, n_tasks))
Y = X @ true_W + rng.normal(scale=0.5, size=(n_samples, n_tasks))

# تدريب MultiTaskLasso
mt_lasso = MultiTaskLasso(alpha=0.3, random_state=42).fit(X, Y)

# فحص الصفوف الصفرية
row_norms = np.linalg.norm(mt_lasso.coef_, axis=1)
active_features = np.where(row_norms > 0)[0]

print(f"الميزات الحقيقية المنشأة: [2, 7, 14, 21]")
print(f"الميزات المنتقاة جماعياً بواسطة MultiTaskLasso: {list(active_features)}")
print(f"أبعاد مصفوفة المعاملات coef_: {mt_lasso.coef_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 25))
# إنشاء ميزتين مكررتين بشدة
X[:, 1] = X[:, 0] + rng.normal(scale=0.01, size=100)

true_W = np.zeros((25, 3))
true_W[0, :] = [3.0, 2.5, 4.0]
true_W[1, :] = [2.0, 3.5, 1.5]
Y = X @ true_W + rng.normal(scale=0.2, size=(100, 3))

mt_enet = MultiTaskElasticNet(alpha=0.2, l1_ratio=0.7, random_state=42)
mt_enet.fit(X, Y)

print("معاملات الميزة 0 عبر الأهداف الثلاثة:", np.round(mt_enet.coef_[0], 3))
print("معاملات الميزة 1 عبر الأهداف الثلاثة:", np.round(mt_enet.coef_[1], 3))
print("كلا الميزتين تم الإبقاء عليهما معاً دون إقصاء تعسفي!")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import enet_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# استخراج مسار المعاملات لـ ElasticNet
alphas, coefs, dual_gaps = enet_path(
    X_scaled, y, l1_ratio=0.8, alphas=np.logspace(0, -3, 30), random_state=42
)

print(f"شكل مصفوفة المسار (الميزات x نقاط alpha): {coefs.shape}")
print(f"أعلى قيمة alpha (تصفير شبه تام): {alphas[0]:.4f}")
print(f"أدنى قيمة alpha (اقتراب من الحل الحر): {alphas[-1]:.4f}")
print(f"عدد الميزات غير الصفرية عند أدنى alpha: {np.sum(coefs[:, -1] != 0)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
X_scaled = np.asarray(StandardScaler().fit_transform(X))
y_arr = np.asarray(y)

# حساب مسار LARS الدقيق هندسياً
alphas, active, coefs = lars_path(X_scaled, y_arr, method="lasso")

print("ترتيب دخول الميزات في مسار LARS التدريجي:")
for step, feat_idx in enumerate(active[:5]):
    print(f"  الخطوة {step+1}: الميزة '{feature_names[feat_idx]}'")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import MultiTaskLasso, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

housing = fetch_california_housing()
X = np.asarray(StandardScaler().fit_transform(housing.data[:500]))
# 3 أهداف متعددة
Y = np.asarray(StandardScaler().fit_transform(
    np.column_stack([housing.target[:500], housing.data[:500, 0], housing.data[:500, 5]])
))

# نموذج متعدد المهام
mt = MultiTaskLasso(alpha=0.1, random_state=42).fit(X, Y)
mt_selected = set(np.where(np.linalg.norm(mt.coef_, axis=1) > 0)[0])

# نماذج منفصلة
sep_selected = set()
for task in range(Y.shape[1]):
    l = Lasso(alpha=0.1, random_state=42).fit(X, Y[:, task])
    sep_selected.update(np.where(l.coef_ != 0)[0])

print(f"الميزات المنتقاة جماعياً بواسطة MultiTaskLasso ({len(mt_selected)}): {mt_selected}")
print(f"اتحاد الميزات المنتقاة عبر النماذج المنفصلة ({len(sep_selected)}): {sep_selected}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch31.md
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.linear_model import lasso_path, lars_path

rng = np.random.default_rng(42)
X = rng.normal(size=(600, 150))
y = X[:, :10].sum(axis=1) + rng.normal(size=600)

t0 = time.perf_counter()
_ = lasso_path(X, y, alphas=np.logspace(0, -3, 40))
t_cd = time.perf_counter() - t0

t0 = time.perf_counter()
_ = lars_path(X, y, method="lasso")
t_lars = time.perf_counter() - t0

print(f"زمن النزول الإحداثي (lasso_path): {t_cd * 1000:.2f} ms")
print(f"زمن الزوايا الأقل (lars_path):    {t_lars * 1000:.2f} ms")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.default_rng(42)
n_samples, n_features, n_tasks = 120, 30, 4

# مصفوفة بيانات مشتركة
X = rng.normal(size=(n_samples, n_features))

# 4 ميزات فقط هي الحقيقية المؤثرة في كل الأهداف
true_W = np.zeros((n_features, n_tasks))
true_W[[2, 7, 14, 21], :] = rng.uniform(2.0, 5.0, size=(4, n_tasks))
Y = X @ true_W + rng.normal(scale=0.5, size=(n_samples, n_tasks))

# تدريب MultiTaskLasso
mt_lasso = MultiTaskLasso(alpha=0.3, random_state=42).fit(X, Y)

# فحص الصفوف الصفرية
row_norms = np.linalg.norm(mt_lasso.coef_, axis=1)
active_features = np.where(row_norms > 0)[0]

print(f"الميزات الحقيقية المنشأة: [2, 7, 14, 21]")
print(f"الميزات المنتقاة جماعياً بواسطة MultiTaskLasso: {list(active_features)}")
print(f"أبعاد مصفوفة المعاملات coef_: {mt_lasso.coef_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 25))
# إنشاء ميزتين مكررتين بشدة
X[:, 1] = X[:, 0] + rng.normal(scale=0.01, size=100)

true_W = np.zeros((25, 3))
true_W[0, :] = [3.0, 2.5, 4.0]
true_W[1, :] = [2.0, 3.5, 1.5]
Y = X @ true_W + rng.normal(scale=0.2, size=(100, 3))

mt_enet = MultiTaskElasticNet(alpha=0.2, l1_ratio=0.7, random_state=42)
mt_enet.fit(X, Y)

print("معاملات الميزة 0 عبر الأهداف الثلاثة:", np.round(mt_enet.coef_[0], 3))
print("معاملات الميزة 1 عبر الأهداف الثلاثة:", np.round(mt_enet.coef_[1], 3))
print("كلا الميزتين تم الإبقاء عليهما معاً دون إقصاء تعسفي!")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import enet_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# استخراج مسار المعاملات لـ ElasticNet
alphas, coefs, dual_gaps = enet_path(
    X_scaled, y, l1_ratio=0.8, alphas=np.logspace(0, -3, 30), random_state=42
)

print(f"شكل مصفوفة المسار (الميزات x نقاط alpha): {coefs.shape}")
print(f"أعلى قيمة alpha (تصفير شبه تام): {alphas[0]:.4f}")
print(f"أدنى قيمة alpha (اقتراب من الحل الحر): {alphas[-1]:.4f}")
print(f"عدد الميزات غير الصفرية عند أدنى alpha: {np.sum(coefs[:, -1] != 0)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
X_scaled = np.asarray(StandardScaler().fit_transform(X))
y_arr = np.asarray(y)

# حساب مسار LARS الدقيق هندسياً
alphas, active, coefs = lars_path(X_scaled, y_arr, method="lasso")

print("ترتيب دخول الميزات في مسار LARS التدريجي:")
for step, feat_idx in enumerate(active[:5]):
    print(f"  الخطوة {step+1}: الميزة '{feature_names[feat_idx]}'")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import MultiTaskLasso, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

housing = fetch_california_housing()
X = np.asarray(StandardScaler().fit_transform(housing.data[:500]))
# 3 أهداف متعددة
Y = np.asarray(StandardScaler().fit_transform(
    np.column_stack([housing.target[:500], housing.data[:500, 0], housing.data[:500, 5]])
))

# نموذج متعدد المهام
mt = MultiTaskLasso(alpha=0.1, random_state=42).fit(X, Y)
mt_selected = set(np.where(np.linalg.norm(mt.coef_, axis=1) > 0)[0])

# نماذج منفصلة
sep_selected = set()
for task in range(Y.shape[1]):
    l = Lasso(alpha=0.1, random_state=42).fit(X, Y[:, task])
    sep_selected.update(np.where(l.coef_ != 0)[0])

print(f"الميزات المنتقاة جماعياً بواسطة MultiTaskLasso ({len(mt_selected)}): {mt_selected}")
print(f"اتحاد الميزات المنتقاة عبر النماذج المنفصلة ({len(sep_selected)}): {sep_selected}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-31-ch31.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.linear_model import lasso_path, lars_path

rng = np.random.default_rng(42)
X = rng.normal(size=(600, 150))
y = X[:, :10].sum(axis=1) + rng.normal(size=600)

t0 = time.perf_counter()
_ = lasso_path(X, y, alphas=np.logspace(0, -3, 40))
t_cd = time.perf_counter() - t0

t0 = time.perf_counter()
_ = lars_path(X, y, method="lasso")
t_lars = time.perf_counter() - t0

print(f"زمن النزول الإحداثي (lasso_path): {t_cd * 1000:.2f} ms")
print(f"زمن الزوايا الأقل (lars_path):    {t_lars * 1000:.2f} ms")
