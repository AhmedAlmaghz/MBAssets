#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — HistGradientBoosting: المعيار الصناعي
Mastering scikit-learn 1.9 — HistGradientBoosting

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-47-ch47 (lesson 47)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-47-ch47.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — HistGradientBoosting: المعيار الصناعي
Mastering scikit-learn 1.9 — HistGradientBoosting

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-47-ch47 (lesson 47)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

print("البارامترات المفتاحية لـ HistGradientBoostingClassifier في 1.9:")
p = HistGradientBoostingClassifier().get_params()
for k in ["max_iter", "max_leaf_nodes", "learning_rate", "max_bins",
          "l2_regularization", "max_features", "categorical_features",
          "early_stopping", "validation_fraction", "n_iter_no_change", "loss"]:
    print(f"  {k} = {p[k]!r}")

X, y = make_classification(n_samples=20_000, n_features=20, n_informative=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

t0 = time.perf_counter()
gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
t_gb = time.perf_counter() - t0
print(f"GradientBoosting | الزمن: {t_gb:6.2f} s | دقة الاختبار: {gb.score(X_te, y_te):.4f}")

t0 = time.perf_counter()
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(X_tr, y_tr)
t_h = time.perf_counter() - t0
print(f"HistGB           | الزمن: {t_h:6.2f} s | دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"عامل السرعة: {t_gb / t_h:.1f}x لصالح HistGB")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=300_000, n_features=20, n_informative=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

t0 = time.perf_counter()
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(X_tr, y_tr)
t = time.perf_counter() - t0
print(f"300,000 عينة × 20 ميزة | الزمن: {t:.1f} s | n_iter_ = {hgb.n_iter_}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=4000, n_features=8, n_informative=4, random_state=42)
rng = np.random.default_rng(42)
mask = rng.random(X.shape) < 0.08
X_nan = X.copy()
X_nan[mask] = np.nan
X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y, test_size=0.3, random_state=42, stratify=y)

try:
    GradientBoostingClassifier().fit(X_tr, y_tr)
except ValueError as err:
    print(f"GradientBoosting يرفض البيانات: {str(err)[:60]}...")

native = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
imp_h = make_pipeline(SimpleImputer(strategy="mean"), HistGradientBoostingClassifier(random_state=42)).fit(X_tr, y_tr)
imp_g = make_pipeline(SimpleImputer(strategy="mean"), GradientBoostingClassifier(random_state=42)).fit(X_tr, y_tr)
print(f"HistGB على NaN مباشرة:       {native.score(X_te, y_te):.4f}")
print(f"Imputer ثم HistGB:           {imp_h.score(X_te, y_te):.4f}")
print(f"Imputer ثم GradientBoosting: {imp_g.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 2000
city = rng.choice(["الرياض", "جدة", "الدمام"], size=n, p=[0.5, 0.3, 0.2])
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
base = 0.8 * x1 - 0.5 * x2
base += np.where(city == "الرياض", 1.2, np.where(city == "جدة", 0.3, -0.8))
y = (base + 0.5 * rng.normal(size=n) > 0).astype(int)

df = pd.DataFrame({"x1": x1, "x2": x2, "city": pd.Categorical(city)})
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

hgb = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"الأعمدة الفئوية المكتشفة تلقائياً: {hgb.is_categorical_}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=3000, n_features=8, n_informative=5, class_sep=1.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

hgb = HistGradientBoostingClassifier(
    max_iter=500, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=15, random_state=42,
).fit(X_tr, y_tr)
print(f"طلبنا 500 تكرار فبنى فعلياً {hgb.n_iter_} فقط")
print(f"أفضل درجة تحقق: {np.max(np.asarray(hgb.validation_score_)):.4f}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import KFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [
    ("HistGB", HistGradientBoostingClassifier(random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(random_state=42, n_jobs=-1)),
]:
    s = cross_val_score(model, X, y, cv=cv).mean()
    print(f"{name:16s} | متوسط دقة 5-fold: {s:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=10, n_informative=5, random_state=42)
rng = np.random.default_rng(42)

for rate in [0.05, 0.15, 0.30]:
    X_nan = X.copy()
    X_nan[rng.random(X.shape) < rate] = np.nan
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nan, y, test_size=0.3, random_state=42, stratify=y
    )
    native = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    imp = make_pipeline(
        SimpleImputer(strategy="mean"),
        HistGradientBoostingClassifier(random_state=42),
    ).fit(X_tr, y_tr)
    diff = native.score(X_te, y_te) - imp.score(X_te, y_te)
    print(f"ضياع {rate:.0%} | NaN مباشرة: {native.score(X_te, y_te):.4f}"
          f" | Imputer: {imp.score(X_te, y_te):.4f} | الفارق: {diff:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part06/ch47.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=15, n_informative=8,
                           class_sep=0.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

best = (0.0, None)
for leaves in [15, 31, 63]:
    for l2 in [0.0, 1.0]:
        hgb = HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, l2_regularization=l2, random_state=42,
        ).fit(X_tr, y_tr)
        s = hgb.score(X_te, y_te)
        print(f"leaves={leaves:2d} | l2={l2} | دقة الاختبار: {s:.4f}")
        if s > best[0]:
            best = (s, (leaves, l2))
print(f"التركيبة الأفضل: {best[1]} بدقة {best[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

print("البارامترات المفتاحية لـ HistGradientBoostingClassifier في 1.9:")
p = HistGradientBoostingClassifier().get_params()
for k in ["max_iter", "max_leaf_nodes", "learning_rate", "max_bins",
          "l2_regularization", "max_features", "categorical_features",
          "early_stopping", "validation_fraction", "n_iter_no_change", "loss"]:
    print(f"  {k} = {p[k]!r}")

X, y = make_classification(n_samples=20_000, n_features=20, n_informative=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

t0 = time.perf_counter()
gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
t_gb = time.perf_counter() - t0
print(f"GradientBoosting | الزمن: {t_gb:6.2f} s | دقة الاختبار: {gb.score(X_te, y_te):.4f}")

t0 = time.perf_counter()
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(X_tr, y_tr)
t_h = time.perf_counter() - t0
print(f"HistGB           | الزمن: {t_h:6.2f} s | دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"عامل السرعة: {t_gb / t_h:.1f}x لصالح HistGB")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=300_000, n_features=20, n_informative=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

t0 = time.perf_counter()
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(X_tr, y_tr)
t = time.perf_counter() - t0
print(f"300,000 عينة × 20 ميزة | الزمن: {t:.1f} s | n_iter_ = {hgb.n_iter_}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=4000, n_features=8, n_informative=4, random_state=42)
rng = np.random.default_rng(42)
mask = rng.random(X.shape) < 0.08
X_nan = X.copy()
X_nan[mask] = np.nan
X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y, test_size=0.3, random_state=42, stratify=y)

try:
    GradientBoostingClassifier().fit(X_tr, y_tr)
except ValueError as err:
    print(f"GradientBoosting يرفض البيانات: {str(err)[:60]}...")

native = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
imp_h = make_pipeline(SimpleImputer(strategy="mean"), HistGradientBoostingClassifier(random_state=42)).fit(X_tr, y_tr)
imp_g = make_pipeline(SimpleImputer(strategy="mean"), GradientBoostingClassifier(random_state=42)).fit(X_tr, y_tr)
print(f"HistGB على NaN مباشرة:       {native.score(X_te, y_te):.4f}")
print(f"Imputer ثم HistGB:           {imp_h.score(X_te, y_te):.4f}")
print(f"Imputer ثم GradientBoosting: {imp_g.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 2000
city = rng.choice(["الرياض", "جدة", "الدمام"], size=n, p=[0.5, 0.3, 0.2])
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
base = 0.8 * x1 - 0.5 * x2
base += np.where(city == "الرياض", 1.2, np.where(city == "جدة", 0.3, -0.8))
y = (base + 0.5 * rng.normal(size=n) > 0).astype(int)

df = pd.DataFrame({"x1": x1, "x2": x2, "city": pd.Categorical(city)})
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

hgb = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")
print(f"الأعمدة الفئوية المكتشفة تلقائياً: {hgb.is_categorical_}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=3000, n_features=8, n_informative=5, class_sep=1.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

hgb = HistGradientBoostingClassifier(
    max_iter=500, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=15, random_state=42,
).fit(X_tr, y_tr)
print(f"طلبنا 500 تكرار فبنى فعلياً {hgb.n_iter_} فقط")
print(f"أفضل درجة تحقق: {np.max(np.asarray(hgb.validation_score_)):.4f}")
print(f"دقة الاختبار: {hgb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import KFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [
    ("HistGB", HistGradientBoostingClassifier(random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(random_state=42, n_jobs=-1)),
]:
    s = cross_val_score(model, X, y, cv=cv).mean()
    print(f"{name:16s} | متوسط دقة 5-fold: {s:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=10, n_informative=5, random_state=42)
rng = np.random.default_rng(42)

for rate in [0.05, 0.15, 0.30]:
    X_nan = X.copy()
    X_nan[rng.random(X.shape) < rate] = np.nan
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nan, y, test_size=0.3, random_state=42, stratify=y
    )
    native = HistGradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    imp = make_pipeline(
        SimpleImputer(strategy="mean"),
        HistGradientBoostingClassifier(random_state=42),
    ).fit(X_tr, y_tr)
    diff = native.score(X_te, y_te) - imp.score(X_te, y_te)
    print(f"ضياع {rate:.0%} | NaN مباشرة: {native.score(X_te, y_te):.4f}"
          f" | Imputer: {imp.score(X_te, y_te):.4f} | الفارق: {diff:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-47-ch47.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=15, n_informative=8,
                           class_sep=0.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

best = (0.0, None)
for leaves in [15, 31, 63]:
    for l2 in [0.0, 1.0]:
        hgb = HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, l2_regularization=l2, random_state=42,
        ).fit(X_tr, y_tr)
        s = hgb.score(X_te, y_te)
        print(f"leaves={leaves:2d} | l2={l2} | دقة الاختبار: {s:.4f}")
        if s > best[0]:
            best = (s, (leaves, l2))
print(f"التركيبة الأفضل: {best[1]} بدقة {best[0]:.4f}")
