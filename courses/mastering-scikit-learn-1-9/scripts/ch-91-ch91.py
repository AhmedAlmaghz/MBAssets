#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — توجيه البيانات الوصفية
Mastering scikit-learn 1.9 — Chapter 91 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-91-ch91 (lesson 91)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-91-ch91.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — توجيه البيانات الوصفية
Mastering scikit-learn 1.9 — Chapter 91 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-91-ch91 (lesson 91)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

w = compute_sample_weight("balanced", y_train)

pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
pipe.fit(X_train, y_train, sample_weight=w)          # من يستلمها؟ المتّبع؟ المصنّف؟
search.fit(X_train, y_train, sample_weight=w)        # ومن يستلمها في المقاييس؟

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

sklearn.set_config(enable_metadata_routing=False)
print("التوجيه مطفأ افتراضياً:", sklearn.get_config()["enable_metadata_routing"])

X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                           weights=[0.9, 0.1], flip_y=0.02, random_state=42)
w = compute_sample_weight("balanced", y)
print("أوزان الصفوف:", np.unique(np.round(w, 4)))

pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
grid = GridSearchCV(pipe, {"clf__C": [0.5, 1.0]}, cv=3, error_score="raise")
try:
    grid.fit(X, y, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
print("بعد التشغيل:", sklearn.get_config()["enable_metadata_routing"])

pipe2 = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
grid2 = GridSearchCV(pipe2, {"clf__C": [0.5, 1.0]}, cv=3, error_score="raise")
try:
    grid2.fit(X, y, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
import inspect

print(inspect.signature(StandardScaler.fit))
# (self, X, y=None, sample_weight=None)

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
pipe2.named_steps["sc"].set_fit_request(sample_weight=True)
pipe2.named_steps["clf"].set_fit_request(sample_weight=True).set_score_request(
    sample_weight=True)
grid2.fit(X, y, sample_weight=w)
print("نجح البحث:", grid2.best_params_, "| الدرجة:", round(grid2.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.metadata_routing import get_routing_for_object

METHODS = ("fit", "score", "split", "transform")
for name, step in pipe2.named_steps.items():
    req = get_routing_for_object(step)
    cells = " | ".join(
        f"{m}={getattr(req, m).requests or '-'}"
        for m in METHODS if hasattr(req, m))
    print(f"{name:<4} {type(req).__name__:<15} {cells}")
print("جسم الأنبوب:", type(pipe2.get_metadata_routing()).__name__)

# ────────────────────────────────────────────────────────────────────────
# [007] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42,
                                  stratify=y)
wa = compute_sample_weight("balanced", ya)
print("موجود في التدريب:", np.bincount(ya), "| في الاختبار:", np.bincount(yb))
print("قيمة الوزن للفئة النادرة:", round(wa[ya == 1][0], 4))


def fresh():
    return Pipeline([("sc", StandardScaler()), ("clf", SVC())])


plain = fresh().fit(Xa, ya)
p_plain = plain.predict(Xb)
weighted = fresh()
weighted.named_steps["sc"].set_fit_request(sample_weight=True)
weighted.named_steps["clf"].set_fit_request(sample_weight=True)
weighted.fit(Xa, ya, sample_weight=wa)
p_wt = weighted.predict(Xb)

for tag, mdl, pred in (("بلا أوزان", plain, p_plain),
                       ("بالأوزان", weighted, p_wt)):
    tp = int(((pred == 1) & (yb == 1)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    fn = int(((pred == 0) & (yb == 1)).sum())
    print(f"{tag:<10} دقة={mdl.score(Xb, yb):.4f} "
          f"استرجاع={recall_score(yb, pred, pos_label=1):.4f} "
          f"موجّهة={precision_score(yb, pred, pos_label=1):.4f} "
          f"tp={tp} fp={fp} fn={fn}")
print("الدقة الموجّهة بلا أوزان:",
      round(average_precision_score(yb, plain.decision_function(Xb)), 4))
print("الدقة الموجّهة بالأوزان :",
      round(average_precision_score(yb, weighted.decision_function(Xb)), 4))

# ────────────────────────────────────────────────────────────────────────
# [008] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer

GRID = {"clf__C": [0.5, 1.0, 4.0], "clf__gamma": ["scale", 0.1]}


def fresh2():
    p = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
    p.named_steps["sc"].set_fit_request(sample_weight=True)
    p.named_steps["clf"].set_fit_request(sample_weight=True)
    return p


sc = make_scorer(recall_score, pos_label=1)
try:
    GridSearchCV(fresh2(), GRID, cv=3, scoring=sc, error_score="raise").fit(
        Xa, ya, sample_weight=wa)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

sc2 = make_scorer(recall_score, pos_label=1)
sc2.set_score_request(sample_weight=True)
g_w = GridSearchCV(fresh2(), GRID, cv=3, scoring=sc2, error_score="raise")
g_w.fit(Xa, ya, sample_weight=wa)
g_p = GridSearchCV(fresh2(), GRID, cv=3, scoring=sc2, error_score="raise")
g_p.fit(Xa, ya)
print("مع أوزان  :", g_w.best_params_, round(g_w.best_score_, 4))
print("بلا أوزان :", g_p.best_params_, round(g_p.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [009] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
sc = make_scorer(recall_score, pos_label=1)
sc.set_score_request(sample_weight=True)   # نفس الكائن، لا كائن آخر
print(type(sc).__name__)

# ────────────────────────────────────────────────────────────────────────
# [010] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
groups = np.repeat(np.arange(60), rng.integers(8, 26, size=60))
Xg, yg = make_classification(n_samples=len(groups), n_features=20,
                             n_informative=8, random_state=42)
from sklearn.model_selection import GroupKFold, KFold

gkf = GroupKFold(n_splits=5)
folds = [set(g) for _, g in gkf.split(Xg, yg, groups)]
print("أحجام الطيّات:", [len(f) for f in folds])
print("تقاطع الطيّة ٠ مع الأولى:", len(folds[0] & folds[1]))

from sklearn.linear_model import LogisticRegression

lin = Pipeline([("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000))])
gk = GridSearchCV(lin, {"clf__C": [0.5, 2.0]}, cv=GroupKFold(5),
                  error_score="raise")
gk.fit(Xg, yg, groups=groups)
print("وصلت المجموعات:", gk.best_params_, round(gk.best_score_, 4))
print("طلب التقسيم:", get_routing_for_object(GroupKFold(5)).split.requests)

try:
    GridSearchCV(lin, {"clf__C": [0.5, 2.0]}, cv=KFold(5),
                 error_score="raise").fit(Xg, yg, groups=groups)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [011] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import cross_validate

wg = compute_sample_weight("balanced", yg)


def lin2():
    p = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
    p.named_steps["sc"].set_fit_request(sample_weight=True)
    p.named_steps["clf"].set_fit_request(sample_weight=True)
    return p


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quiet = cross_validate(lin2(), Xg, yg, cv=GroupKFold(5),
                           params={"sample_weight": wg, "groups": groups})
print("بلا score_request:", np.round(quiet["test_score"], 4))
print("نوع التحذير:", [w.category.__name__ for w in caught][:1])


def lin3():
    p = lin2()
    p.named_steps["clf"].set_score_request(sample_weight=True)
    return p


loud = cross_validate(lin3(), Xg, yg, cv=GroupKFold(5),
                      params={"sample_weight": wg, "groups": groups},
                      error_score="raise")
print("مع score_request:", np.round(loud["test_score"], 4))
plain_cv = cross_validate(lin2(), Xg, yg, cv=GroupKFold(5),
                          params={"groups": groups}, error_score="raise")
print("متوسط موزون:", round(loud["test_score"].mean(), 4),
      "| متوسط غير موزون:", round(plain_cv["test_score"].mean(), 4))

# ────────────────────────────────────────────────────────────────────────
# [012] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42,
                                  stratify=y)
w = compute_sample_weight("balanced", ya)

pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
pipe.named_steps["clf"].set_fit_request(sample_weight=True)
try:
    pipe.fit(Xa, ya, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [013] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.metadata_routing import get_routing_for_object

print("المفتاح:", sklearn.get_config()["enable_metadata_routing"])

Xi, yi = make_classification(n_samples=600, n_features=20, n_informative=6,
                             random_state=42)
miss = np.random.default_rng(0).choice(600, size=90, replace=False)
Xi[miss, 4] = np.nan
Xi[miss, 11] = np.nan
Xa, Xb, ya, yb = train_test_split(Xi, yi, test_size=0.25, random_state=42,
                                  stratify=yi)

pipe3 = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
pipe3.named_steps["clf"].set_fit_request(sample_weight=True)
pipe3.named_steps["clf"].set_score_request(sample_weight=True)

METHODS = ("fit", "score", "transform")
for name, step in pipe3.named_steps.items():
    req = get_routing_for_object(step)
    cells = " | ".join(f"{m}={getattr(req, m).requests or '-'}"
                       for m in METHODS if hasattr(req, m))
    print(f"{name:<4} {cells}")

clf = pipe3.named_steps["clf"]
sc = StandardScaler().fit(Xa)
print("المصفوفة نفسها:", np.array_equal(sc.transform(Xa), sc.transform(Xb)))

# ────────────────────────────────────────────────────────────────────────
# [014] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     StratifiedKFold)

rng = np.random.default_rng(42)
sizes = rng.integers(8, 26, size=60)
grp = np.repeat(np.arange(60), sizes)
Xm, ym = make_classification(n_samples=len(grp), n_features=20,
                             n_informative=8, random_state=42)

folds = [set(g) for _, g in GroupKFold(4).split(Xm, ym, grp)]
print("أحجام الطيّات:", [len(f) for f in folds])
print("تداخل الطيّتين الأوليين:", len(folds[0] & folds[1]))
print("كل مجموعة في طيّة واحدة فقط:",
      all(len(f) == 60 // 4 or len(f) in (60 // 4, 60 // 4 + 1)
          for f in folds))

lin2 = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=5000))])
g2 = GridSearchCV(lin2, {"clf__C": [0.5, 2.0]}, cv=GroupKFold(4),
                  error_score="raise")
g2.fit(Xm, ym, groups=grp)
print("أفضل تركيبة:", g2.best_params_, "| الدرجة:", round(g2.best_score_, 4))

try:
    GridSearchCV(lin2, {"clf__C": [0.5, 2.0]}, cv=StratifiedKFold(4),
                 error_score="raise").fit(Xm, ym, groups=grp)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [015] content/part11/ch91.md
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
Xi2, yi2 = make_classification(n_samples=500, n_features=15,
                              weights=[0.85, 0.15], random_state=42)
Xa2, Xb2, ya2, yb2 = train_test_split(Xi2, yi2, test_size=0.25,
                                      random_state=42, stratify=yi2)
wi2 = compute_sample_weight("balanced", ya2)
w_full = compute_sample_weight("balanced", yi2)

bad = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
bad.named_steps["sc"].set_fit_request(sample_weight=False)
bad.named_steps["clf"].set_fit_request(sample_weight=True)
bad.fit(Xa2, ya2, sample_weight=wi2)
print("fit نجح، درجة:", round(bad.score(Xb2, yb2), 4))

res = cross_validate(bad, Xi2, yi2, cv=3, params={"sample_weight": w_full})
print("درجات التحقق:", np.round(res["test_score"], 4))

good = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
good.named_steps["sc"].set_fit_request(sample_weight=False)
good.named_steps["clf"].set_fit_request(sample_weight=True)
good.named_steps["clf"].set_score_request(sample_weight=True)
res2 = cross_validate(good, Xi2, yi2, cv=3, params={"sample_weight": w_full},
                      error_score="raise")
print("بعد التصحيح:", np.round(res2["test_score"], 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

w = compute_sample_weight("balanced", y_train)

pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
pipe.fit(X_train, y_train, sample_weight=w)          # من يستلمها؟ المتّبع؟ المصنّف؟
search.fit(X_train, y_train, sample_weight=w)        # ومن يستلمها في المقاييس؟

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

sklearn.set_config(enable_metadata_routing=False)
print("التوجيه مطفأ افتراضياً:", sklearn.get_config()["enable_metadata_routing"])

X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                           weights=[0.9, 0.1], flip_y=0.02, random_state=42)
w = compute_sample_weight("balanced", y)
print("أوزان الصفوف:", np.unique(np.round(w, 4)))

pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
grid = GridSearchCV(pipe, {"clf__C": [0.5, 1.0]}, cv=3, error_score="raise")
try:
    grid.fit(X, y, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
print("بعد التشغيل:", sklearn.get_config()["enable_metadata_routing"])

pipe2 = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
grid2 = GridSearchCV(pipe2, {"clf__C": [0.5, 1.0]}, cv=3, error_score="raise")
try:
    grid2.fit(X, y, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
import inspect

print(inspect.signature(StandardScaler.fit))
# (self, X, y=None, sample_weight=None)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
pipe2.named_steps["sc"].set_fit_request(sample_weight=True)
pipe2.named_steps["clf"].set_fit_request(sample_weight=True).set_score_request(
    sample_weight=True)
grid2.fit(X, y, sample_weight=w)
print("نجح البحث:", grid2.best_params_, "| الدرجة:", round(grid2.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.metadata_routing import get_routing_for_object

METHODS = ("fit", "score", "split", "transform")
for name, step in pipe2.named_steps.items():
    req = get_routing_for_object(step)
    cells = " | ".join(
        f"{m}={getattr(req, m).requests or '-'}"
        for m in METHODS if hasattr(req, m))
    print(f"{name:<4} {type(req).__name__:<15} {cells}")
print("جسم الأنبوب:", type(pipe2.get_metadata_routing()).__name__)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42,
                                  stratify=y)
wa = compute_sample_weight("balanced", ya)
print("موجود في التدريب:", np.bincount(ya), "| في الاختبار:", np.bincount(yb))
print("قيمة الوزن للفئة النادرة:", round(wa[ya == 1][0], 4))


def fresh():
    return Pipeline([("sc", StandardScaler()), ("clf", SVC())])


plain = fresh().fit(Xa, ya)
p_plain = plain.predict(Xb)
weighted = fresh()
weighted.named_steps["sc"].set_fit_request(sample_weight=True)
weighted.named_steps["clf"].set_fit_request(sample_weight=True)
weighted.fit(Xa, ya, sample_weight=wa)
p_wt = weighted.predict(Xb)

for tag, mdl, pred in (("بلا أوزان", plain, p_plain),
                       ("بالأوزان", weighted, p_wt)):
    tp = int(((pred == 1) & (yb == 1)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    fn = int(((pred == 0) & (yb == 1)).sum())
    print(f"{tag:<10} دقة={mdl.score(Xb, yb):.4f} "
          f"استرجاع={recall_score(yb, pred, pos_label=1):.4f} "
          f"موجّهة={precision_score(yb, pred, pos_label=1):.4f} "
          f"tp={tp} fp={fp} fn={fn}")
print("الدقة الموجّهة بلا أوزان:",
      round(average_precision_score(yb, plain.decision_function(Xb)), 4))
print("الدقة الموجّهة بالأوزان :",
      round(average_precision_score(yb, weighted.decision_function(Xb)), 4))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer

GRID = {"clf__C": [0.5, 1.0, 4.0], "clf__gamma": ["scale", 0.1]}


def fresh2():
    p = Pipeline([("sc", StandardScaler()), ("clf", SVC())])
    p.named_steps["sc"].set_fit_request(sample_weight=True)
    p.named_steps["clf"].set_fit_request(sample_weight=True)
    return p


sc = make_scorer(recall_score, pos_label=1)
try:
    GridSearchCV(fresh2(), GRID, cv=3, scoring=sc, error_score="raise").fit(
        Xa, ya, sample_weight=wa)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

sc2 = make_scorer(recall_score, pos_label=1)
sc2.set_score_request(sample_weight=True)
g_w = GridSearchCV(fresh2(), GRID, cv=3, scoring=sc2, error_score="raise")
g_w.fit(Xa, ya, sample_weight=wa)
g_p = GridSearchCV(fresh2(), GRID, cv=3, scoring=sc2, error_score="raise")
g_p.fit(Xa, ya)
print("مع أوزان  :", g_w.best_params_, round(g_w.best_score_, 4))
print("بلا أوزان :", g_p.best_params_, round(g_p.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
sc = make_scorer(recall_score, pos_label=1)
sc.set_score_request(sample_weight=True)   # نفس الكائن، لا كائن آخر
print(type(sc).__name__)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
groups = np.repeat(np.arange(60), rng.integers(8, 26, size=60))
Xg, yg = make_classification(n_samples=len(groups), n_features=20,
                             n_informative=8, random_state=42)
from sklearn.model_selection import GroupKFold, KFold

gkf = GroupKFold(n_splits=5)
folds = [set(g) for _, g in gkf.split(Xg, yg, groups)]
print("أحجام الطيّات:", [len(f) for f in folds])
print("تقاطع الطيّة ٠ مع الأولى:", len(folds[0] & folds[1]))

from sklearn.linear_model import LogisticRegression

lin = Pipeline([("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000))])
gk = GridSearchCV(lin, {"clf__C": [0.5, 2.0]}, cv=GroupKFold(5),
                  error_score="raise")
gk.fit(Xg, yg, groups=groups)
print("وصلت المجموعات:", gk.best_params_, round(gk.best_score_, 4))
print("طلب التقسيم:", get_routing_for_object(GroupKFold(5)).split.requests)

try:
    GridSearchCV(lin, {"clf__C": [0.5, 2.0]}, cv=KFold(5),
                 error_score="raise").fit(Xg, yg, groups=groups)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import cross_validate

wg = compute_sample_weight("balanced", yg)


def lin2():
    p = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
    p.named_steps["sc"].set_fit_request(sample_weight=True)
    p.named_steps["clf"].set_fit_request(sample_weight=True)
    return p


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quiet = cross_validate(lin2(), Xg, yg, cv=GroupKFold(5),
                           params={"sample_weight": wg, "groups": groups})
print("بلا score_request:", np.round(quiet["test_score"], 4))
print("نوع التحذير:", [w.category.__name__ for w in caught][:1])


def lin3():
    p = lin2()
    p.named_steps["clf"].set_score_request(sample_weight=True)
    return p


loud = cross_validate(lin3(), Xg, yg, cv=GroupKFold(5),
                      params={"sample_weight": wg, "groups": groups},
                      error_score="raise")
print("مع score_request:", np.round(loud["test_score"], 4))
plain_cv = cross_validate(lin2(), Xg, yg, cv=GroupKFold(5),
                          params={"groups": groups}, error_score="raise")
print("متوسط موزون:", round(loud["test_score"].mean(), 4),
      "| متوسط غير موزون:", round(plain_cv["test_score"].mean(), 4))

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42,
                                  stratify=y)
w = compute_sample_weight("balanced", ya)

pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
pipe.named_steps["clf"].set_fit_request(sample_weight=True)
try:
    pipe.fit(Xa, ya, sample_weight=w)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.metadata_routing import get_routing_for_object

print("المفتاح:", sklearn.get_config()["enable_metadata_routing"])

Xi, yi = make_classification(n_samples=600, n_features=20, n_informative=6,
                             random_state=42)
miss = np.random.default_rng(0).choice(600, size=90, replace=False)
Xi[miss, 4] = np.nan
Xi[miss, 11] = np.nan
Xa, Xb, ya, yb = train_test_split(Xi, yi, test_size=0.25, random_state=42,
                                  stratify=yi)

pipe3 = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
pipe3.named_steps["clf"].set_fit_request(sample_weight=True)
pipe3.named_steps["clf"].set_score_request(sample_weight=True)

METHODS = ("fit", "score", "transform")
for name, step in pipe3.named_steps.items():
    req = get_routing_for_object(step)
    cells = " | ".join(f"{m}={getattr(req, m).requests or '-'}"
                       for m in METHODS if hasattr(req, m))
    print(f"{name:<4} {cells}")

clf = pipe3.named_steps["clf"]
sc = StandardScaler().fit(Xa)
print("المصفوفة نفسها:", np.array_equal(sc.transform(Xa), sc.transform(Xb)))

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     StratifiedKFold)

rng = np.random.default_rng(42)
sizes = rng.integers(8, 26, size=60)
grp = np.repeat(np.arange(60), sizes)
Xm, ym = make_classification(n_samples=len(grp), n_features=20,
                             n_informative=8, random_state=42)

folds = [set(g) for _, g in GroupKFold(4).split(Xm, ym, grp)]
print("أحجام الطيّات:", [len(f) for f in folds])
print("تداخل الطيّتين الأوليين:", len(folds[0] & folds[1]))
print("كل مجموعة في طيّة واحدة فقط:",
      all(len(f) == 60 // 4 or len(f) in (60 // 4, 60 // 4 + 1)
          for f in folds))

lin2 = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=5000))])
g2 = GridSearchCV(lin2, {"clf__C": [0.5, 2.0]}, cv=GroupKFold(4),
                  error_score="raise")
g2.fit(Xm, ym, groups=grp)
print("أفضل تركيبة:", g2.best_params_, "| الدرجة:", round(g2.best_score_, 4))

try:
    GridSearchCV(lin2, {"clf__C": [0.5, 2.0]}, cv=StratifiedKFold(4),
                 error_score="raise").fit(Xm, ym, groups=grp)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-91-ch91.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
sklearn.set_config(enable_metadata_routing=True)
Xi2, yi2 = make_classification(n_samples=500, n_features=15,
                              weights=[0.85, 0.15], random_state=42)
Xa2, Xb2, ya2, yb2 = train_test_split(Xi2, yi2, test_size=0.25,
                                      random_state=42, stratify=yi2)
wi2 = compute_sample_weight("balanced", ya2)
w_full = compute_sample_weight("balanced", yi2)

bad = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
bad.named_steps["sc"].set_fit_request(sample_weight=False)
bad.named_steps["clf"].set_fit_request(sample_weight=True)
bad.fit(Xa2, ya2, sample_weight=wi2)
print("fit نجح، درجة:", round(bad.score(Xb2, yb2), 4))

res = cross_validate(bad, Xi2, yi2, cv=3, params={"sample_weight": w_full})
print("درجات التحقق:", np.round(res["test_score"], 4))

good = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])
good.named_steps["sc"].set_fit_request(sample_weight=False)
good.named_steps["clf"].set_fit_request(sample_weight=True)
good.named_steps["clf"].set_score_request(sample_weight=True)
res2 = cross_validate(good, Xi2, yi2, cv=3, params={"sample_weight": w_full},
                      error_score="raise")
print("بعد التصحيح:", np.round(res2["test_score"], 4))
