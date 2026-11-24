#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — البحث العشوائي والتوزيعات
Mastering scikit-learn 1.9 — Chapter 82 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-82-ch82 (lesson 82)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-82-ch82.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — البحث العشوائي والتوزيعات
Mastering scikit-learn 1.9 — Chapter 82 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-82-ch82 (lesson 82)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.stats import loguniform, uniform, randint, reciprocal

print("loguniform(1e-3, 1e1):", np.round(loguniform(1e-3, 1e1).rvs(6, random_state=42), 5))
print("reciprocal(0.1, 100):  ", np.round(reciprocal(0.1, 100).rvs(5, random_state=42), 2))
print("randint(2, 10):        ", randint(2, 10).rvs(5, random_state=42))
print("uniform(0, 1):         ", np.round(uniform(0, 1).rvs(5, random_state=42), 3))

# أين تقع العينات؟ الفوارق المطلقة مقابل النسبية
vals = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
print("الفوارق المطلقة بين القيم:", np.round(np.diff(vals), 4))
print("النسب بين القيم المتتالية:", np.round(vals[1:] / vals[:-1], 1))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
X, y = X[:300], y[:300]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

# نداء تمهيدي صغير يهيئ عمال التوازي قبل التجربة الرئيسة
GridSearchCV(pipe, {"svc__C": [1]}, cv=2, n_jobs=-1).fit(X_tr[:200], y_tr[:200])

param_grid = {"svc__gamma": [0.0005, 0.001, 0.002, 0.005, 0.01],
              "svc__degree": [2, 3, 4],
              "svc__coef0": [0.0, 1.0],
              "svc__tol": [1e-3, 1e-2]}
n_combos = 5 * 3 * 2 * 2
t0 = time.perf_counter()
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
t_grid = time.perf_counter() - t0

print(f"التركيبات: {n_combos} → تدريبات: {n_combos * 5}")
print(f"قيم gamma فريدة في المساحة كلها: 5 فقط — نسبة الهدر {(n_combos - 5) / n_combos:.1%}")
print(f"أفضل درجة تحقق: {grid.best_score_:.4f} عند gamma={grid.best_params_['svc__gamma']} | الزمن: {t_grid:.1f} s")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import loguniform, uniform, randint
from sklearn.model_selection import RandomizedSearchCV

t0 = time.perf_counter()
rand = RandomizedSearchCV(
    pipe,
    {"svc__gamma": loguniform(5e-4, 5e-2),
     "svc__degree": randint(2, 5),
     "svc__coef0": uniform(0, 1),
     "svc__tol": uniform(1e-4, 9e-3)},
    n_iter=20, cv=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rand = time.perf_counter() - t0

n_distinct = len(set(np.round(rand.cv_results_["param_svc__gamma"].astype(float), 7)))
print(f"التدريبات: 100 فقط | قيم gamma فريدة جُرِّبت: {n_distinct}")
print(f"أفضل درجة تحقق: {rand.best_score_:.4f} عند gamma={rand.best_params_['svc__gamma']:.5f}")
print(f"على الاختبار المحجوز — الشبكة: {grid.score(X_te, y_te):.4f} | العشوائي: {rand.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.experimental.enable_halving_search_cv import (  # تفعيل المسار التجريبي والاستيراد معاً
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
import pandas as pd

t0 = time.perf_counter()
hal = HalvingGridSearchCV(pipe, param_grid, factor=3, min_resources=50,
                          cv=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_hal = time.perf_counter() - t0

hres = pd.DataFrame(hal.cv_results_)
summary = hres.groupby("iter").agg(
    candidates=("mean_test_score", "size"),
    n_samples=("n_resources", "first"),
    best_score=("mean_test_score", "max"))
print(summary.to_string())
print(f"أفضل درجة: {hal.best_score_:.4f} عند gamma={hal.best_params_['svc__gamma']} | الزمن: {t_hal:.1f} s")
print(f"عيّنات مستهلكة: {int(hres['n_resources'].sum())} مقابل {60 * 5 * 180} للشبكة الكاملة")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
g0 = float(rand.best_params_["svc__gamma"])
fine_grid = {"svc__gamma": np.geomspace(g0 / 4, g0 * 4, 9)}
t0 = time.perf_counter()
fine = GridSearchCV(pipe, fine_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
t_fine = time.perf_counter() - t0
print(f"المرحلة الدقيقة: 9 تركيبات حول gamma={g0:.4f} → درجة {fine.best_score_:.4f} في {t_fine:.1f} s")

comparison = pd.DataFrame({
    "الاستراتيجية": ["شبكة كاملة", "عشوائي (n_iter=20)", "مرحلتان: عشوائي + شبكة دقيقة", "Halving"],
    "تدريبات": [300, 100, 145, 400],
    "أفضل درجة تحقق": [grid.best_score_, rand.best_score_, fine.best_score_, hal.best_score_],
    "زمن (ث)": [t_grid, t_rand, t_rand + t_fine, t_hal],
})
print(comparison.to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
lin_grid = {"svc__gamma": np.linspace(1e-4, 10, 10)}
geo_grid = {"svc__gamma": np.geomspace(1e-4, 10, 10)}
print("قيم الشبكة الخطية:", np.round(np.linspace(1e-4, 10, 10), 3))

t0 = time.perf_counter()
bad = GridSearchCV(pipe, lin_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
good = GridSearchCV(pipe, geo_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
print(f"خطية: أفضل درجة {bad.best_score_:.4f} عند gamma={bad.best_params_['svc__gamma']:.4f}")
print(f"هندسية: أفضل درجة {good.best_score_:.4f} عند gamma={good.best_params_['svc__gamma']:.5f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

Xw, yw = load_wine(return_X_y=True)
rf_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
rs = RandomizedSearchCV(
    rf_pipe,
    {"randomforestclassifier__n_estimators": randint(50, 300),
     "randomforestclassifier__max_features": uniform(0.1, 0.9),
     "randomforestclassifier__min_samples_leaf": randint(1, 8)},
    n_iter=25, cv=5, random_state=42, n_jobs=-1).fit(Xw, yw)

baseline = cross_val_score(
    make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    Xw, yw, cv=5, n_jobs=-1).mean()

print(f"أفضل درجة بعد 25 تجربة: {rs.best_score_:.4f}")
print(f"أفضل تركيبة: n_estimators={rs.best_params_['randomforestclassifier__n_estimators']}, "
      f"max_features={rs.best_params_['randomforestclassifier__max_features']:.2f}, "
      f"min_samples_leaf={rs.best_params_['randomforestclassifier__min_samples_leaf']}")
print(f"الغابة الافتراضية: {baseline:.4f} | التحسن: {rs.best_score_ - baseline:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch82.md
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import loguniform, uniform

edges = 10.0 ** np.arange(-4, 3)   # 1e-4 حتى 1e2
draws_log = loguniform(1e-4, 1e2).rvs(1000, random_state=42)
draws_lin = uniform(1e-4, 1e2).rvs(1000, random_state=42)

for name, draws in [("loguniform", draws_log), ("uniform", draws_lin)]:
    counts = np.histogram(draws, bins=edges)[0]
    shares = counts / counts.sum()
    print(name, "لكل عقد:", np.round(shares, 3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.stats import loguniform, uniform, randint, reciprocal

print("loguniform(1e-3, 1e1):", np.round(loguniform(1e-3, 1e1).rvs(6, random_state=42), 5))
print("reciprocal(0.1, 100):  ", np.round(reciprocal(0.1, 100).rvs(5, random_state=42), 2))
print("randint(2, 10):        ", randint(2, 10).rvs(5, random_state=42))
print("uniform(0, 1):         ", np.round(uniform(0, 1).rvs(5, random_state=42), 3))

# أين تقع العينات؟ الفوارق المطلقة مقابل النسبية
vals = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
print("الفوارق المطلقة بين القيم:", np.round(np.diff(vals), 4))
print("النسب بين القيم المتتالية:", np.round(vals[1:] / vals[:-1], 1))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
X, y = X[:300], y[:300]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

# نداء تمهيدي صغير يهيئ عمال التوازي قبل التجربة الرئيسة
GridSearchCV(pipe, {"svc__C": [1]}, cv=2, n_jobs=-1).fit(X_tr[:200], y_tr[:200])

param_grid = {"svc__gamma": [0.0005, 0.001, 0.002, 0.005, 0.01],
              "svc__degree": [2, 3, 4],
              "svc__coef0": [0.0, 1.0],
              "svc__tol": [1e-3, 1e-2]}
n_combos = 5 * 3 * 2 * 2
t0 = time.perf_counter()
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
t_grid = time.perf_counter() - t0

print(f"التركيبات: {n_combos} → تدريبات: {n_combos * 5}")
print(f"قيم gamma فريدة في المساحة كلها: 5 فقط — نسبة الهدر {(n_combos - 5) / n_combos:.1%}")
print(f"أفضل درجة تحقق: {grid.best_score_:.4f} عند gamma={grid.best_params_['svc__gamma']} | الزمن: {t_grid:.1f} s")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import loguniform, uniform, randint
from sklearn.model_selection import RandomizedSearchCV

t0 = time.perf_counter()
rand = RandomizedSearchCV(
    pipe,
    {"svc__gamma": loguniform(5e-4, 5e-2),
     "svc__degree": randint(2, 5),
     "svc__coef0": uniform(0, 1),
     "svc__tol": uniform(1e-4, 9e-3)},
    n_iter=20, cv=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rand = time.perf_counter() - t0

n_distinct = len(set(np.round(rand.cv_results_["param_svc__gamma"].astype(float), 7)))
print(f"التدريبات: 100 فقط | قيم gamma فريدة جُرِّبت: {n_distinct}")
print(f"أفضل درجة تحقق: {rand.best_score_:.4f} عند gamma={rand.best_params_['svc__gamma']:.5f}")
print(f"على الاختبار المحجوز — الشبكة: {grid.score(X_te, y_te):.4f} | العشوائي: {rand.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.experimental.enable_halving_search_cv import (  # تفعيل المسار التجريبي والاستيراد معاً
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
import pandas as pd

t0 = time.perf_counter()
hal = HalvingGridSearchCV(pipe, param_grid, factor=3, min_resources=50,
                          cv=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_hal = time.perf_counter() - t0

hres = pd.DataFrame(hal.cv_results_)
summary = hres.groupby("iter").agg(
    candidates=("mean_test_score", "size"),
    n_samples=("n_resources", "first"),
    best_score=("mean_test_score", "max"))
print(summary.to_string())
print(f"أفضل درجة: {hal.best_score_:.4f} عند gamma={hal.best_params_['svc__gamma']} | الزمن: {t_hal:.1f} s")
print(f"عيّنات مستهلكة: {int(hres['n_resources'].sum())} مقابل {60 * 5 * 180} للشبكة الكاملة")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
g0 = float(rand.best_params_["svc__gamma"])
fine_grid = {"svc__gamma": np.geomspace(g0 / 4, g0 * 4, 9)}
t0 = time.perf_counter()
fine = GridSearchCV(pipe, fine_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
t_fine = time.perf_counter() - t0
print(f"المرحلة الدقيقة: 9 تركيبات حول gamma={g0:.4f} → درجة {fine.best_score_:.4f} في {t_fine:.1f} s")

comparison = pd.DataFrame({
    "الاستراتيجية": ["شبكة كاملة", "عشوائي (n_iter=20)", "مرحلتان: عشوائي + شبكة دقيقة", "Halving"],
    "تدريبات": [300, 100, 145, 400],
    "أفضل درجة تحقق": [grid.best_score_, rand.best_score_, fine.best_score_, hal.best_score_],
    "زمن (ث)": [t_grid, t_rand, t_rand + t_fine, t_hal],
})
print(comparison.to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
lin_grid = {"svc__gamma": np.linspace(1e-4, 10, 10)}
geo_grid = {"svc__gamma": np.geomspace(1e-4, 10, 10)}
print("قيم الشبكة الخطية:", np.round(np.linspace(1e-4, 10, 10), 3))

t0 = time.perf_counter()
bad = GridSearchCV(pipe, lin_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
good = GridSearchCV(pipe, geo_grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
print(f"خطية: أفضل درجة {bad.best_score_:.4f} عند gamma={bad.best_params_['svc__gamma']:.4f}")
print(f"هندسية: أفضل درجة {good.best_score_:.4f} عند gamma={good.best_params_['svc__gamma']:.5f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

Xw, yw = load_wine(return_X_y=True)
rf_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
rs = RandomizedSearchCV(
    rf_pipe,
    {"randomforestclassifier__n_estimators": randint(50, 300),
     "randomforestclassifier__max_features": uniform(0.1, 0.9),
     "randomforestclassifier__min_samples_leaf": randint(1, 8)},
    n_iter=25, cv=5, random_state=42, n_jobs=-1).fit(Xw, yw)

baseline = cross_val_score(
    make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    Xw, yw, cv=5, n_jobs=-1).mean()

print(f"أفضل درجة بعد 25 تجربة: {rs.best_score_:.4f}")
print(f"أفضل تركيبة: n_estimators={rs.best_params_['randomforestclassifier__n_estimators']}, "
      f"max_features={rs.best_params_['randomforestclassifier__max_features']:.2f}, "
      f"min_samples_leaf={rs.best_params_['randomforestclassifier__min_samples_leaf']}")
print(f"الغابة الافتراضية: {baseline:.4f} | التحسن: {rs.best_score_ - baseline:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-82-ch82.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import loguniform, uniform

edges = 10.0 ** np.arange(-4, 3)   # 1e-4 حتى 1e2
draws_log = loguniform(1e-4, 1e2).rvs(1000, random_state=42)
draws_lin = uniform(1e-4, 1e2).rvs(1000, random_state=42)

for name, draws in [("loguniform", draws_log), ("uniform", draws_lin)]:
    counts = np.histogram(draws, bins=edges)[0]
    shares = counts / counts.sum()
    print(name, "لكل عقد:", np.round(shares, 3))
