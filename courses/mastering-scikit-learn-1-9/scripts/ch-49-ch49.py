#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط نموذج لمجموعة كاملة
Mastering scikit-learn 1.9 — Chapter 49 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-49-ch49 (lesson 49)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-49-ch49.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط نموذج لمجموعة كاملة
Mastering scikit-learn 1.9 — Chapter 49 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-49-ch49 (lesson 49)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

configs = [
    ("الافتراضي (كل العينات والميزات)", dict(max_samples=1.0, max_features=1.0)),
    ("نصف العينات فقط",                dict(max_samples=0.5, max_features=1.0)),
    ("نصف الميزات فقط",                dict(max_samples=1.0, max_features=0.5)),
    ("نصف الاثنين معاً",                dict(max_samples=0.5, max_features=0.5)),
]
for name, kw in configs:
    t0 = time.perf_counter()
    b = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=60,
                          bootstrap=True, random_state=42, n_jobs=-1, **kw).fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"{name:32s} | دقة: {b.score(X_te, y_te):.4f} | زمن التدريب: {dt:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_classification(n_samples=900, n_features=14, n_informative=6,
                           n_redundant=3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

for lr, n, d, sub in [(0.5, 60, 2, 1.0), (0.1, 300, 2, 1.0), (0.05, 600, 3, 0.8)]:
    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr,
                                    max_depth=d, subsample=sub, random_state=42)
    t0 = time.perf_counter()
    gb.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"lr={lr:<4} n={n:<4} depth={d} sub={sub} | دقة: {gb.score(X_te, y_te):.4f} | زمن: {dt:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
                                     max_leaf_nodes=31, early_stopping=True,
                                     validation_fraction=0.15, n_iter_no_change=15,
                                     random_state=42)
hgb.fit(X_tr, y_tr)
print(f"طلبت max_iter=500 فتوقف وحده عند n_iter_ = {hgb.n_iter_}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"التكرارات الموفرة: {500 - hgb.n_iter_} ({(500 - hgb.n_iter_) / 500:.0%} من الزمن)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

Xr, yr = make_regression(n_samples=600, n_features=14, n_informative=5,
                         noise=25.0, random_state=42)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.3,
                               max_depth=3, random_state=42).fit(Xr_tr, yr_tr)

staged = np.asarray([r2_score(yr_te, p) for p in gb.staged_predict(Xr_te)])
best_stage = int(np.argmax(staged)) + 1
print(f"القمة عند المرحلة {best_stage} من أصل 400 بدقة R² = {staged[best_stage-1]:.4f}")
print(f"المرحلة الأخيرة (400) بدقة R² = {staged[-1]:.4f} — أقل من القمة!")
print(f"R² على التدريب في النهاية: {gb.score(Xr_tr, yr_tr):.4f} (حفظ كامل للضجيج)")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

X2_tr, X2_te, y2_tr, y2_te = X_tr, X_te, y_tr, y_te
rf = RandomForestClassifier(n_estimators=20, warm_start=True, oob_score=True,
                            random_state=42, n_jobs=-1)
for n in [20, 50, 100, 200, 400]:
    rf.n_estimators = n
    rf.fit(X2_tr, y2_tr)  # يضيف (n - السابق) أشجاراً فقط
    print(f"أشجار: {n:3d} | OOB: {rf.oob_score_:.4f} | دقة الاختبار: {rf.score(X2_te, y2_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

grid = {"max_depth": [6, None], "min_samples_leaf": [1, 5]}
n_fits = 3 * 2 * 2 + 1   # cv × |Θ| + refit
print(f"الشبكة تقتضي {n_fits} تدريباً كاملاً")
gs = GridSearchCV(RandomForestClassifier(n_estimators=40, random_state=42),
                  grid, cv=3)
t0 = time.perf_counter()
gs.fit(X_tr, y_tr)
dt = time.perf_counter() - t0
print(f"الأفضل: {gs.best_params_} | متوسط CV: {gs.best_score_:.4f} | الزمن: {dt:.2f}s")
print(f"دقة الاختبار للفائز: {gs.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar

grid2 = {"n_estimators": [60, 120], "max_depth": [10, None],
         "min_samples_split": [2, 8]}
n_fits2 = 3 * 2 * 2 * 2 + 1
print(f"هذا البحث يخفي {n_fits2} تدريباً كاملاً — وشريط التقدم سيكشفها")
gs2 = GridSearchCV(RandomForestClassifier(random_state=42), grid2, cv=3)
gs2.set_callbacks(ProgressBar(max_propagation_depth=0))
gs2.fit(X_tr, y_tr)
print(f"الفائز: {gs2.best_params_} | CV: {gs2.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3,
                                              random_state=42, stratify=yd)

for lr in [0.5, 0.2, 0.1, 0.05]:
    h = HistGradientBoostingClassifier(max_iter=400, learning_rate=lr,
                                       early_stopping=True, n_iter_no_change=15,
                                       random_state=42).fit(Xd_tr, yd_tr)
    print(f"lr={lr:<5} توقف عند n_iter_={h.n_iter_:3d} | دقة الاختبار: {h.score(Xd_te, yd_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part06/ch49.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
Xc, yc = data.data[:2000], data.target[:2000]
Xc_tr, yc_tr = Xc[:1400], yc[:1400]

rfr = RandomForestRegressor(n_estimators=10, warm_start=True, oob_score=True,
                            random_state=42, n_jobs=-1)
prev = None
for n in [10, 30, 60, 100, 200]:
    rfr.n_estimators = n
    rfr.fit(Xc_tr, yc_tr)
    gain = "—" if prev is None else f"{rfr.oob_score_ - prev:+.4f}"
    print(f"أشجار: {n:3d} | OOB R²: {rfr.oob_score_:.4f} | التحسن: {gain}")
    prev = rfr.oob_score_

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

configs = [
    ("الافتراضي (كل العينات والميزات)", dict(max_samples=1.0, max_features=1.0)),
    ("نصف العينات فقط",                dict(max_samples=0.5, max_features=1.0)),
    ("نصف الميزات فقط",                dict(max_samples=1.0, max_features=0.5)),
    ("نصف الاثنين معاً",                dict(max_samples=0.5, max_features=0.5)),
]
for name, kw in configs:
    t0 = time.perf_counter()
    b = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=60,
                          bootstrap=True, random_state=42, n_jobs=-1, **kw).fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"{name:32s} | دقة: {b.score(X_te, y_te):.4f} | زمن التدريب: {dt:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_classification(n_samples=900, n_features=14, n_informative=6,
                           n_redundant=3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

for lr, n, d, sub in [(0.5, 60, 2, 1.0), (0.1, 300, 2, 1.0), (0.05, 600, 3, 0.8)]:
    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr,
                                    max_depth=d, subsample=sub, random_state=42)
    t0 = time.perf_counter()
    gb.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    print(f"lr={lr:<4} n={n:<4} depth={d} sub={sub} | دقة: {gb.score(X_te, y_te):.4f} | زمن: {dt:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
                                     max_leaf_nodes=31, early_stopping=True,
                                     validation_fraction=0.15, n_iter_no_change=15,
                                     random_state=42)
hgb.fit(X_tr, y_tr)
print(f"طلبت max_iter=500 فتوقف وحده عند n_iter_ = {hgb.n_iter_}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"التكرارات الموفرة: {500 - hgb.n_iter_} ({(500 - hgb.n_iter_) / 500:.0%} من الزمن)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

Xr, yr = make_regression(n_samples=600, n_features=14, n_informative=5,
                         noise=25.0, random_state=42)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.3,
                               max_depth=3, random_state=42).fit(Xr_tr, yr_tr)

staged = np.asarray([r2_score(yr_te, p) for p in gb.staged_predict(Xr_te)])
best_stage = int(np.argmax(staged)) + 1
print(f"القمة عند المرحلة {best_stage} من أصل 400 بدقة R² = {staged[best_stage-1]:.4f}")
print(f"المرحلة الأخيرة (400) بدقة R² = {staged[-1]:.4f} — أقل من القمة!")
print(f"R² على التدريب في النهاية: {gb.score(Xr_tr, yr_tr):.4f} (حفظ كامل للضجيج)")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

X2_tr, X2_te, y2_tr, y2_te = X_tr, X_te, y_tr, y_te
rf = RandomForestClassifier(n_estimators=20, warm_start=True, oob_score=True,
                            random_state=42, n_jobs=-1)
for n in [20, 50, 100, 200, 400]:
    rf.n_estimators = n
    rf.fit(X2_tr, y2_tr)  # يضيف (n - السابق) أشجاراً فقط
    print(f"أشجار: {n:3d} | OOB: {rf.oob_score_:.4f} | دقة الاختبار: {rf.score(X2_te, y2_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

grid = {"max_depth": [6, None], "min_samples_leaf": [1, 5]}
n_fits = 3 * 2 * 2 + 1   # cv × |Θ| + refit
print(f"الشبكة تقتضي {n_fits} تدريباً كاملاً")
gs = GridSearchCV(RandomForestClassifier(n_estimators=40, random_state=42),
                  grid, cv=3)
t0 = time.perf_counter()
gs.fit(X_tr, y_tr)
dt = time.perf_counter() - t0
print(f"الأفضل: {gs.best_params_} | متوسط CV: {gs.best_score_:.4f} | الزمن: {dt:.2f}s")
print(f"دقة الاختبار للفائز: {gs.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar

grid2 = {"n_estimators": [60, 120], "max_depth": [10, None],
         "min_samples_split": [2, 8]}
n_fits2 = 3 * 2 * 2 * 2 + 1
print(f"هذا البحث يخفي {n_fits2} تدريباً كاملاً — وشريط التقدم سيكشفها")
gs2 = GridSearchCV(RandomForestClassifier(random_state=42), grid2, cv=3)
gs2.set_callbacks(ProgressBar(max_propagation_depth=0))
gs2.fit(X_tr, y_tr)
print(f"الفائز: {gs2.best_params_} | CV: {gs2.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3,
                                              random_state=42, stratify=yd)

for lr in [0.5, 0.2, 0.1, 0.05]:
    h = HistGradientBoostingClassifier(max_iter=400, learning_rate=lr,
                                       early_stopping=True, n_iter_no_change=15,
                                       random_state=42).fit(Xd_tr, yd_tr)
    print(f"lr={lr:<5} توقف عند n_iter_={h.n_iter_:3d} | دقة الاختبار: {h.score(Xd_te, yd_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-49-ch49.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
Xc, yc = data.data[:2000], data.target[:2000]
Xc_tr, yc_tr = Xc[:1400], yc[:1400]

rfr = RandomForestRegressor(n_estimators=10, warm_start=True, oob_score=True,
                            random_state=42, n_jobs=-1)
prev = None
for n in [10, 30, 60, 100, 200]:
    rfr.n_estimators = n
    rfr.fit(Xc_tr, yc_tr)
    gain = "—" if prev is None else f"{rfr.oob_score_ - prev:+.4f}"
    print(f"أشجار: {n:3d} | OOB R²: {rfr.oob_score_:.4f} | التحسن: {gain}")
    prev = rfr.oob_score_
