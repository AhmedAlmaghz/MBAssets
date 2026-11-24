#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المنظّم: Ridge و ElasticNet
Mastering scikit-learn 1.9 — Ridge

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-29-ch29 (lesson 29)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-29-ch29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار المنظّم: Ridge و ElasticNet
Mastering scikit-learn 1.9 — Ridge

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-29-ch29 (lesson 29)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# إنشاء بيانات ذات تعدد خطي وخصائص غير أساسية
X, y = make_regression(n_samples=120, n_features=60, n_informative=15,
                       noise=20.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# مقارنة بين OLS غير المقيد و Ridge
pipe_ols = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
pipe_ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=100.0))])

pipe_ols.fit(X_train, y_train)
pipe_ridge.fit(X_train, y_train)

print(f"OLS   - R² التدريب: {pipe_ols.score(X_train, y_train):.4f} | R² الاختبار: {pipe_ols.score(X_test, y_test):.4f}")
print(f"Ridge - R² التدريب: {pipe_ridge.score(X_train, y_train):.4f} | R² الاختبار: {pipe_ridge.score(X_test, y_test):.4f}")
print(f"معيار المعاملات L2 (OLS):   {np.linalg.norm(pipe_ols.named_steps['reg'].coef_):.2f}")
print(f"معيار المعاملات L2 (Ridge): {np.linalg.norm(pipe_ridge.named_steps['reg'].coef_):.2f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing()
X, y = housing.data, housing.target

alphas = np.logspace(-3, 3, 20)
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge_cv", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error"))
])

pipe_cv.fit(X, y)
best_ridge = pipe_cv.named_steps["ridge_cv"]

print(f"أفضل قيمة alpha مختارة: {best_ridge.alpha_:.4f}")
print(f"أفضل معامل تحديد R²: {pipe_cv.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# بيانات تحوي ميزات مكررة ومرتبطة للغاية
X, y = make_regression(n_samples=150, n_features=20, n_informative=5,
                       effective_rank=4, noise=10.0, random_state=42)

pipe_enet = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(alpha=0.5, l1_ratio=0.6, random_state=42))
])

pipe_enet.fit(X, y)
coefs = pipe_enet.named_steps["enet"].coef_

n_zeros = np.sum(coefs == 0.0)
print(f"إجمالي الميزات: {len(coefs)}")
print(f"الميزات المصفرة تماماً بفعل L1: {n_zeros}")
print(f"الميزات المحتفظ بها بفضل التوازن الشبكي: {len(coefs) - n_zeros}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_diabetes(return_X_y=True)

# ضبط l1_ratio و alphas معاً
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
pipe_enet_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("enet_cv", ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=42, alphas=50))
])

pipe_enet_cv.fit(X, y)
best_model = pipe_enet_cv.named_steps["enet_cv"]

print(f"أفضل نسبة l1_ratio مختارة: {best_model.l1_ratio_}")
print(f"أفضل قيمة alpha مختارة: {best_model.alpha_:.4f}")
print(f"عدد الميزات غير الصفرية: {sum(best_model.coef_ != 0)} من أصل {X.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_diabetes(return_X_y=True)

pipe_ols = make_pipeline(StandardScaler(), LinearRegression())
pipe_ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))

score_ols = cross_val_score(pipe_ols, X, y, cv=5, scoring="r2").mean()
score_ridge = cross_val_score(pipe_ridge, X, y, cv=5, scoring="r2").mean()

print(f"متوسط R² لـ OLS:     {score_ols:.4f}")
print(f"متوسط R² لـ RidgeCV: {score_ridge:.4f}")
print(f"فرق التحسن:          {score_ridge - score_ols:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch29.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n_samples = 100
# توليد ميزة مشتركة وإضافة ضجيج خفيف لإنتاج تداخل هائل
base_feat = rng.normal(size=(n_samples, 1))
redundant_block = base_feat + rng.normal(scale=0.05, size=(n_samples, 5))
other_features = rng.normal(size=(n_samples, 75))
X = np.hstack([redundant_block, other_features])
y = redundant_block.sum(axis=1) * 3.0 + rng.normal(scale=1.0, size=n_samples)

X_scaled = StandardScaler().fit_transform(X)

ridge = Ridge(alpha=10.0).fit(X_scaled, y)
enet = ElasticNet(alpha=0.5, l1_ratio=0.8, random_state=42).fit(X_scaled, y)

print("معاملات الميزات الخمس المترابطة في Ridge:")
print(np.round(ridge.coef_[:5], 3))
print("معاملات الميزات الخمس المترابطة في ElasticNet:")
print(np.round(enet.coef_[:5], 3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# إنشاء بيانات ذات تعدد خطي وخصائص غير أساسية
X, y = make_regression(n_samples=120, n_features=60, n_informative=15,
                       noise=20.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# مقارنة بين OLS غير المقيد و Ridge
pipe_ols = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
pipe_ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=100.0))])

pipe_ols.fit(X_train, y_train)
pipe_ridge.fit(X_train, y_train)

print(f"OLS   - R² التدريب: {pipe_ols.score(X_train, y_train):.4f} | R² الاختبار: {pipe_ols.score(X_test, y_test):.4f}")
print(f"Ridge - R² التدريب: {pipe_ridge.score(X_train, y_train):.4f} | R² الاختبار: {pipe_ridge.score(X_test, y_test):.4f}")
print(f"معيار المعاملات L2 (OLS):   {np.linalg.norm(pipe_ols.named_steps['reg'].coef_):.2f}")
print(f"معيار المعاملات L2 (Ridge): {np.linalg.norm(pipe_ridge.named_steps['reg'].coef_):.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing()
X, y = housing.data, housing.target

alphas = np.logspace(-3, 3, 20)
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge_cv", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error"))
])

pipe_cv.fit(X, y)
best_ridge = pipe_cv.named_steps["ridge_cv"]

print(f"أفضل قيمة alpha مختارة: {best_ridge.alpha_:.4f}")
print(f"أفضل معامل تحديد R²: {pipe_cv.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# بيانات تحوي ميزات مكررة ومرتبطة للغاية
X, y = make_regression(n_samples=150, n_features=20, n_informative=5,
                       effective_rank=4, noise=10.0, random_state=42)

pipe_enet = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(alpha=0.5, l1_ratio=0.6, random_state=42))
])

pipe_enet.fit(X, y)
coefs = pipe_enet.named_steps["enet"].coef_

n_zeros = np.sum(coefs == 0.0)
print(f"إجمالي الميزات: {len(coefs)}")
print(f"الميزات المصفرة تماماً بفعل L1: {n_zeros}")
print(f"الميزات المحتفظ بها بفضل التوازن الشبكي: {len(coefs) - n_zeros}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_diabetes(return_X_y=True)

# ضبط l1_ratio و alphas معاً
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
pipe_enet_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("enet_cv", ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=42, alphas=50))
])

pipe_enet_cv.fit(X, y)
best_model = pipe_enet_cv.named_steps["enet_cv"]

print(f"أفضل نسبة l1_ratio مختارة: {best_model.l1_ratio_}")
print(f"أفضل قيمة alpha مختارة: {best_model.alpha_:.4f}")
print(f"عدد الميزات غير الصفرية: {sum(best_model.coef_ != 0)} من أصل {X.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_diabetes(return_X_y=True)

pipe_ols = make_pipeline(StandardScaler(), LinearRegression())
pipe_ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))

score_ols = cross_val_score(pipe_ols, X, y, cv=5, scoring="r2").mean()
score_ridge = cross_val_score(pipe_ridge, X, y, cv=5, scoring="r2").mean()

print(f"متوسط R² لـ OLS:     {score_ols:.4f}")
print(f"متوسط R² لـ RidgeCV: {score_ridge:.4f}")
print(f"فرق التحسن:          {score_ridge - score_ols:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-29-ch29.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n_samples = 100
# توليد ميزة مشتركة وإضافة ضجيج خفيف لإنتاج تداخل هائل
base_feat = rng.normal(size=(n_samples, 1))
redundant_block = base_feat + rng.normal(scale=0.05, size=(n_samples, 5))
other_features = rng.normal(size=(n_samples, 75))
X = np.hstack([redundant_block, other_features])
y = redundant_block.sum(axis=1) * 3.0 + rng.normal(scale=1.0, size=n_samples)

X_scaled = StandardScaler().fit_transform(X)

ridge = Ridge(alpha=10.0).fit(X_scaled, y)
enet = ElasticNet(alpha=0.5, l1_ratio=0.8, random_state=42).fit(X_scaled, y)

print("معاملات الميزات الخمس المترابطة في Ridge:")
print(np.round(ridge.coef_[:5], 3))
print("معاملات الميزات الخمس المترابطة في ElasticNet:")
print(np.round(enet.coef_[:5], 3))
