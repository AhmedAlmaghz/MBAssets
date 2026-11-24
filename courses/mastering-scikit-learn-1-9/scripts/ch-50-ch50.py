#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — متى تستخدم أي مجموعة؟
Mastering scikit-learn 1.9 — Chapter 50 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-50-ch50 (lesson 50)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-50-ch50.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — متى تستخدم أي مجموعة؟
Mastering scikit-learn 1.9 — Chapter 50 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-50-ch50 (lesson 50)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

contenders = [
    ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
    ("شجرة واحدة",          DecisionTreeClassifier(random_state=42)),
    ("Bagging",            BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                             n_estimators=60, random_state=42, n_jobs=-1)),
    ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("ExtraTrees",         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("AdaBoost",           AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("GradientBoosting",   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                      max_depth=2, random_state=42)),
    ("HistGB",             HistGradientBoostingClassifier(max_iter=150, random_state=42)),
]
for name, m in contenders:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    print(f"{name:18s} | دقة: {m.score(X_te, y_te):.4f} | تدريب: {fit_t:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X, y = data.data[:3000], data.target[:3000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

contenders = [
    ("Ridge (خطي)",      make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("شجرة واحدة",        DecisionTreeRegressor(random_state=42)),
    ("Bagging",          BaggingRegressor(DecisionTreeRegressor(random_state=42),
                                          n_estimators=60, random_state=42, n_jobs=-1)),
    ("RandomForest",     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("ExtraTrees",       ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                max_depth=3, random_state=42)),
    ("HistGB",           HistGradientBoostingRegressor(max_iter=200, random_state=42)),
]
for name, m in contenders:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    print(f"{name:18s} | RMSE: {root_mean_squared_error(y_te, m.predict(X_te)):.4f} | تدريب: {fit_t:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = make_classification(n_samples=600, n_features=12, n_informative=4,
                           n_redundant=3, random_state=42)
X_tr0, X_te, y_tr0, y_te = train_test_split(X, y, test_size=0.3,
                                            random_state=42, stratify=y)
rng = np.random.default_rng(42)
accs_tree, accs_bag = [], []
for rep in range(20):
    idx = rng.choice(len(X_tr0), size=int(0.6 * len(X_tr0)), replace=False)
    Xr, yr = X_tr0[idx], y_tr0[idx]
    accs_tree.append(DecisionTreeClassifier(random_state=42).fit(Xr, yr).score(X_te, y_te))
    accs_bag.append(BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                      n_estimators=50, random_state=42,
                                      n_jobs=-1).fit(Xr, yr).score(X_te, y_te))

at, ab = np.asarray(accs_tree), np.asarray(accs_bag)
print(f"شجرة مفردة: متوسط {at.mean():.4f} | انحراف {at.std():.4f} | المدى [{at.min():.4f}, {at.max():.4f}]")
print(f"Bagging 50 : متوسط {ab.mean():.4f} | انحراف {ab.std():.4f} | المدى [{ab.min():.4f}, {ab.max():.4f}]")
print(f"نسبة خفض الانحراف: {1 - ab.std() / at.std():.0%}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

Xf, yf = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=42)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.3, random_state=42)

lin = LinearRegression().fit(Xf_tr, yf_tr)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2,
                               random_state=42).fit(Xf_tr, yf_tr)
print(f"R² للانحدار الخطي:  {r2_score(yf_te, lin.predict(Xf_te)):.4f} (سقف انحيازه)")
print(f"R² لتعزيز التدرج:   {r2_score(yf_te, gb.predict(Xf_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
models = [
    ("شجرة واحدة",   DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("ExtraTrees",   ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("HistGB",       HistGradientBoostingClassifier(max_iter=150, random_state=42)),
]
for name, m in models:
    m.fit(X_tr, y_tr)
    m.predict(X_te)  # تسخين قبل القياس
    t0 = time.perf_counter()
    for _ in range(20):
        m.predict(X_te)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"{name:14s} | دقة: {m.score(X_te, y_te):.4f} | زمن استدلال 171 عينة: {dt:.2f} ms")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# القطب الأول: بيانات صغيرة نظيفة (Iris)
Xi, yi = load_iris(return_X_y=True)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3,
                                              random_state=42, stratify=yi)
gs_i = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [2, 3, 4], "min_samples_leaf": [1, 3, 5]}, cv=5)
gs_i.fit(Xi_tr, yi_tr)
rf_i = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xi_tr, yi_tr)
print(f"Iris    | شجرة مضبوطة {gs_i.best_params_} | دقة: {gs_i.score(Xi_te, yi_te):.4f} | عقد: {gs_i.best_estimator_.tree_.node_count}")
print(f"Iris    | غابة افتراضية 100 شجرة      | دقة: {rf_i.score(Xi_te, yi_te):.4f}")

# القطب الثاني: بيانات أوسع وأصعب (سرطان الثدي)
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3,
                                              random_state=42, stratify=yc)
gs_c = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}, cv=5)
gs_c.fit(Xc_tr, yc_tr)
rf_c = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xc_tr, yc_tr)
print(f"Cancer  | شجرة مضبوطة | CV: {gs_c.best_score_:.4f} | اختبار: {gs_c.score(Xc_te, yc_te):.4f}")
print(f"Cancer  | غابة افتراضية | اختبار: {rf_c.score(Xc_te, yc_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

single = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
bagged = make_pipeline(StandardScaler(),
                       BaggingClassifier(LogisticRegression(max_iter=5000),
                                         n_estimators=60, random_state=42, n_jobs=-1))
single.fit(X_tr, y_tr)
bagged.fit(X_tr, y_tr)
print(f"اللوجيستي المفرد:  {single.score(X_te, y_te):.4f}")
print(f"مجمّع على 60 نسخة: {bagged.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part06/ch50.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X, y = data.data[:5000], data.target[:5000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for name, m in [("RandomForest", RandomForestRegressor(n_estimators=100,
                                                       random_state=42, n_jobs=-1)),
                ("HistGB", HistGradientBoostingRegressor(max_iter=300,
                                                         random_state=42))]:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"{name:14s} | RMSE: {root_mean_squared_error(y_te, m.predict(X_te)):.4f} | تدريب: {dt:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

contenders = [
    ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
    ("شجرة واحدة",          DecisionTreeClassifier(random_state=42)),
    ("Bagging",            BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                             n_estimators=60, random_state=42, n_jobs=-1)),
    ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("ExtraTrees",         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("AdaBoost",           AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("GradientBoosting",   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                      max_depth=2, random_state=42)),
    ("HistGB",             HistGradientBoostingClassifier(max_iter=150, random_state=42)),
]
for name, m in contenders:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    print(f"{name:18s} | دقة: {m.score(X_te, y_te):.4f} | تدريب: {fit_t:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X, y = data.data[:3000], data.target[:3000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

contenders = [
    ("Ridge (خطي)",      make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("شجرة واحدة",        DecisionTreeRegressor(random_state=42)),
    ("Bagging",          BaggingRegressor(DecisionTreeRegressor(random_state=42),
                                          n_estimators=60, random_state=42, n_jobs=-1)),
    ("RandomForest",     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("ExtraTrees",       ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                max_depth=3, random_state=42)),
    ("HistGB",           HistGradientBoostingRegressor(max_iter=200, random_state=42)),
]
for name, m in contenders:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    print(f"{name:18s} | RMSE: {root_mean_squared_error(y_te, m.predict(X_te)):.4f} | تدريب: {fit_t:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = make_classification(n_samples=600, n_features=12, n_informative=4,
                           n_redundant=3, random_state=42)
X_tr0, X_te, y_tr0, y_te = train_test_split(X, y, test_size=0.3,
                                            random_state=42, stratify=y)
rng = np.random.default_rng(42)
accs_tree, accs_bag = [], []
for rep in range(20):
    idx = rng.choice(len(X_tr0), size=int(0.6 * len(X_tr0)), replace=False)
    Xr, yr = X_tr0[idx], y_tr0[idx]
    accs_tree.append(DecisionTreeClassifier(random_state=42).fit(Xr, yr).score(X_te, y_te))
    accs_bag.append(BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                      n_estimators=50, random_state=42,
                                      n_jobs=-1).fit(Xr, yr).score(X_te, y_te))

at, ab = np.asarray(accs_tree), np.asarray(accs_bag)
print(f"شجرة مفردة: متوسط {at.mean():.4f} | انحراف {at.std():.4f} | المدى [{at.min():.4f}, {at.max():.4f}]")
print(f"Bagging 50 : متوسط {ab.mean():.4f} | انحراف {ab.std():.4f} | المدى [{ab.min():.4f}, {ab.max():.4f}]")
print(f"نسبة خفض الانحراف: {1 - ab.std() / at.std():.0%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

Xf, yf = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=42)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.3, random_state=42)

lin = LinearRegression().fit(Xf_tr, yf_tr)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2,
                               random_state=42).fit(Xf_tr, yf_tr)
print(f"R² للانحدار الخطي:  {r2_score(yf_te, lin.predict(Xf_te)):.4f} (سقف انحيازه)")
print(f"R² لتعزيز التدرج:   {r2_score(yf_te, gb.predict(Xf_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
models = [
    ("شجرة واحدة",   DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("ExtraTrees",   ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("HistGB",       HistGradientBoostingClassifier(max_iter=150, random_state=42)),
]
for name, m in models:
    m.fit(X_tr, y_tr)
    m.predict(X_te)  # تسخين قبل القياس
    t0 = time.perf_counter()
    for _ in range(20):
        m.predict(X_te)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"{name:14s} | دقة: {m.score(X_te, y_te):.4f} | زمن استدلال 171 عينة: {dt:.2f} ms")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# القطب الأول: بيانات صغيرة نظيفة (Iris)
Xi, yi = load_iris(return_X_y=True)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3,
                                              random_state=42, stratify=yi)
gs_i = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [2, 3, 4], "min_samples_leaf": [1, 3, 5]}, cv=5)
gs_i.fit(Xi_tr, yi_tr)
rf_i = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xi_tr, yi_tr)
print(f"Iris    | شجرة مضبوطة {gs_i.best_params_} | دقة: {gs_i.score(Xi_te, yi_te):.4f} | عقد: {gs_i.best_estimator_.tree_.node_count}")
print(f"Iris    | غابة افتراضية 100 شجرة      | دقة: {rf_i.score(Xi_te, yi_te):.4f}")

# القطب الثاني: بيانات أوسع وأصعب (سرطان الثدي)
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3,
                                              random_state=42, stratify=yc)
gs_c = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}, cv=5)
gs_c.fit(Xc_tr, yc_tr)
rf_c = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xc_tr, yc_tr)
print(f"Cancer  | شجرة مضبوطة | CV: {gs_c.best_score_:.4f} | اختبار: {gs_c.score(Xc_te, yc_te):.4f}")
print(f"Cancer  | غابة افتراضية | اختبار: {rf_c.score(Xc_te, yc_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

single = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
bagged = make_pipeline(StandardScaler(),
                       BaggingClassifier(LogisticRegression(max_iter=5000),
                                         n_estimators=60, random_state=42, n_jobs=-1))
single.fit(X_tr, y_tr)
bagged.fit(X_tr, y_tr)
print(f"اللوجيستي المفرد:  {single.score(X_te, y_te):.4f}")
print(f"مجمّع على 60 نسخة: {bagged.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-50-ch50.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X, y = data.data[:5000], data.target[:5000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for name, m in [("RandomForest", RandomForestRegressor(n_estimators=100,
                                                       random_state=42, n_jobs=-1)),
                ("HistGB", HistGradientBoostingRegressor(max_iter=300,
                                                         random_state=42))]:
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"{name:14s} | RMSE: {root_mean_squared_error(y_te, m.predict(X_te)):.4f} | تدريب: {dt:.2f}s")
