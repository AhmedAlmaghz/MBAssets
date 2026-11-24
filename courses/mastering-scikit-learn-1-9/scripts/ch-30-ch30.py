#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — Lasso: النموذج الخطي الذي يُصفّر
Mastering scikit-learn 1.9 — Lasso

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-30-ch30 (lesson 30)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-30-ch30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — Lasso: النموذج الخطي الذي يُصفّر
Mastering scikit-learn 1.9 — Lasso

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-30-ch30 (lesson 30)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch30.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# إنشاء بيانات تحوي 50 ميزة، 8 منها فقط دالة حقيقة
X, y = make_regression(n_samples=150, n_features=50, n_informative=8,
                       noise=15.0, random_state=42)

pipe_ols = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
pipe_lasso = Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=2.5, random_state=42))])

pipe_ols.fit(X, y)
pipe_lasso.fit(X, y)

coef_ols = pipe_ols.named_steps["reg"].coef_
coef_lasso = pipe_lasso.named_steps["reg"].coef_

print(f"إجمالي الميزات في البيانات: {X.shape[1]}")
print(f"الميزات الصفرية في OLS:     {np.sum(coef_ols == 0)}")
print(f"الميزات الصفرية في Lasso:   {np.sum(coef_lasso == 0)}")
print(f"الميزات المنتقاة في Lasso:  {np.sum(coef_lasso != 0)}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch30.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
X_scaled = StandardScaler().fit_transform(X)

# استخراج مسار المعاملات بتناقص alpha
alphas, coefs_path, _ = lasso_path(X_scaled, y, alphas=np.logspace(1, -2, 50))

print(f"عدد قيم alpha المختبرة في المسار: {len(alphas)}")
# فحص أول ميزة انضمت للمسار (أول ميزة غير صفرية عند أعلى alpha)
first_nonzero_step = np.argmax(np.sum(coefs_path != 0, axis=0) > 0)
first_active_feature = feature_names[np.argmax(np.abs(coefs_path[:, first_nonzero_step]))]
print(f"أول ميزة اخترقت الصفر عند أعلى قيمة لعقوبة alpha: {first_active_feature}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch30.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = np.array(load_diabetes().feature_names)

pipe_lasso_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso_cv", LassoCV(cv=5, random_state=42, alphas=100))
])

pipe_lasso_cv.fit(X, y)
best_lasso = pipe_lasso_cv.named_steps["lasso_cv"]

selected_mask = best_lasso.coef_ != 0
print(f"أفضل قيمة alpha مختارة بالتحقق المتقاطع: {best_lasso.alpha_:.4f}")
print(f"عدد الميزات النشطة: {np.sum(selected_mask)} من أصل {len(feature_names)}")
print(f"الميزات الأساسية المحتفظ بها: {list(feature_names[selected_mask])}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch30.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
names = np.array(data.feature_names)

X_sc = StandardScaler().fit_transform(X)

for alpha in [0.1, 0.01]:
    model = Lasso(alpha=alpha, random_state=42).fit(X_sc, y)
    active = names[model.coef_ != 0]
    zeroed = names[model.coef_ == 0]
    print(f"عند alpha = {alpha}:")
    print(f"  الميزات المتبقية ({len(active)}): {list(active)}")
    print(f"  الميزات المصفرة ({len(zeroed)}): {list(zeroed)}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch30.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = make_regression(n_samples=200, n_features=100, n_informative=10,
                       noise=15.0, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

pipe_ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
pipe_lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42)).fit(X_tr, y_tr)

r2_ols = pipe_ols.score(X_te, y_te)
r2_lasso = pipe_lasso.score(X_te, y_te)
active_lasso = np.sum(pipe_lasso.named_steps["lassocv"].coef_ != 0)

print(f"OLS - R² الاختبار:     {r2_ols:.4f} (جميع الميزات 100 مستخدمة)")
print(f"LassoCV - R² الاختبار: {r2_lasso:.4f} (الميزات المستخدمة: {active_lasso})")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-30-ch30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# إنشاء بيانات تحوي 50 ميزة، 8 منها فقط دالة حقيقة
X, y = make_regression(n_samples=150, n_features=50, n_informative=8,
                       noise=15.0, random_state=42)

pipe_ols = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
pipe_lasso = Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=2.5, random_state=42))])

pipe_ols.fit(X, y)
pipe_lasso.fit(X, y)

coef_ols = pipe_ols.named_steps["reg"].coef_
coef_lasso = pipe_lasso.named_steps["reg"].coef_

print(f"إجمالي الميزات في البيانات: {X.shape[1]}")
print(f"الميزات الصفرية في OLS:     {np.sum(coef_ols == 0)}")
print(f"الميزات الصفرية في Lasso:   {np.sum(coef_lasso == 0)}")
print(f"الميزات المنتقاة في Lasso:  {np.sum(coef_lasso != 0)}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-30-ch30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
X_scaled = StandardScaler().fit_transform(X)

# استخراج مسار المعاملات بتناقص alpha
alphas, coefs_path, _ = lasso_path(X_scaled, y, alphas=np.logspace(1, -2, 50))

print(f"عدد قيم alpha المختبرة في المسار: {len(alphas)}")
# فحص أول ميزة انضمت للمسار (أول ميزة غير صفرية عند أعلى alpha)
first_nonzero_step = np.argmax(np.sum(coefs_path != 0, axis=0) > 0)
first_active_feature = feature_names[np.argmax(np.abs(coefs_path[:, first_nonzero_step]))]
print(f"أول ميزة اخترقت الصفر عند أعلى قيمة لعقوبة alpha: {first_active_feature}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-30-ch30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_diabetes(return_X_y=True)
feature_names = np.array(load_diabetes().feature_names)

pipe_lasso_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso_cv", LassoCV(cv=5, random_state=42, alphas=100))
])

pipe_lasso_cv.fit(X, y)
best_lasso = pipe_lasso_cv.named_steps["lasso_cv"]

selected_mask = best_lasso.coef_ != 0
print(f"أفضل قيمة alpha مختارة بالتحقق المتقاطع: {best_lasso.alpha_:.4f}")
print(f"عدد الميزات النشطة: {np.sum(selected_mask)} من أصل {len(feature_names)}")
print(f"الميزات الأساسية المحتفظ بها: {list(feature_names[selected_mask])}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-30-ch30.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
names = np.array(data.feature_names)

X_sc = StandardScaler().fit_transform(X)

for alpha in [0.1, 0.01]:
    model = Lasso(alpha=alpha, random_state=42).fit(X_sc, y)
    active = names[model.coef_ != 0]
    zeroed = names[model.coef_ == 0]
    print(f"عند alpha = {alpha}:")
    print(f"  الميزات المتبقية ({len(active)}): {list(active)}")
    print(f"  الميزات المصفرة ({len(zeroed)}): {list(zeroed)}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-30-ch30.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = make_regression(n_samples=200, n_features=100, n_informative=10,
                       noise=15.0, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

pipe_ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
pipe_lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42)).fit(X_tr, y_tr)

r2_ols = pipe_ols.score(X_te, y_te)
r2_lasso = pipe_lasso.score(X_te, y_te)
active_lasso = np.sum(pipe_lasso.named_steps["lassocv"].coef_ != 0)

print(f"OLS - R² الاختبار:     {r2_ols:.4f} (جميع الميزات 100 مستخدمة)")
print(f"LassoCV - R² الاختبار: {r2_lasso:.4f} (الميزات المستخدمة: {active_lasso})")
