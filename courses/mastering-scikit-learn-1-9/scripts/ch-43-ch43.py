#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الغابة العشوائية
Mastering scikit-learn 1.9 — Chapter 43 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-43-ch43 (lesson 43)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-43-ch43.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الغابة العشوائية
Mastering scikit-learn 1.9 — Chapter 43 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-43-ch43 (lesson 43)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

# 1. شجرة قرار مفردة (عالية التباين)
tree = DecisionTreeClassifier(random_state=42)
score_tree = cross_val_score(tree, X, y, cv=5).mean()

# 2. غابة عشوائية مكونة من 100 شجرة
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
score_rf = cross_val_score(rf, X, y, cv=5).mean()

print(f"دقة شجرة القرار المفردة: {score_tree:.4f}")
print(f"دقة الغابة العشوائية (100 شجرة): {score_rf:.4f}")
print(f"مكسب التعميم التجميعي: {score_rf - score_tree:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

housing = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    housing.data[:1500], housing.target[:1500], test_size=0.3, random_state=42
)

# اختبار تأثير max_features على خطأ الانحدار
for mf in [1.0, "sqrt", 0.33]:
    rf_reg = RandomForestRegressor(
        n_estimators=60, max_features=mf, random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, rf_reg.predict(X_te))
    print(f"معامل max_features={str(mf):<6} | RMSE الاختبار: {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time

X, y = load_breast_cancer(return_X_y=True)

# مقارنة زمن التدريب والدقة بين RandomForest و ExtraTrees
t0 = time.perf_counter()
rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X, y)
t_rf = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
et = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X, y)
t_et = (time.perf_counter() - t0) * 1000

print(f"RandomForest | الدقة: {rf.score(X, y):.4f} | الزمن: {t_rf:.2f} ms")
print(f"ExtraTrees   | الدقة: {et.score(X, y):.4f} | الزمن: {t_et:.2f} ms")
print(f"ExtraTrees أسرع بنسبة: {(t_rf - t_et) / t_rf:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)

# تفعيل oob_score=True ومراقبة الاستقرار
for n_trees in [10, 30, 70, 150]:
    forest = RandomForestClassifier(
        n_estimators=n_trees, oob_score=True, random_state=42, n_jobs=-1
    ).fit(X, y)
    print(f"أشجار: {n_trees:3d} | دقة التدريب: {forest.score(X, y):.4f} | درجة OOB: {forest.oob_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import time

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

# RandomForest
t0 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rf = time.perf_counter() - t0
rmse_rf = root_mean_squared_error(y_te, rf.predict(X_te))

# ExtraTrees
t0 = time.perf_counter()
et = ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_et = time.perf_counter() - t0
rmse_et = root_mean_squared_error(y_te, et.predict(X_te))

print(f"RandomForest | RMSE: {rmse_rf:.4f} | زمن التدريب: {t_rf * 1000:.1f} ms")
print(f"ExtraTrees   | RMSE: {rmse_et:.4f} | زمن التدريب: {t_et * 1000:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch43.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_iris(return_X_y=True)
forest = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

# عينة اختبار حدودية بين فئتين
ambiguous_sample = np.array([[6.0, 2.7, 5.1, 1.6]])

# جمع توقعات جميع الأشجار الـ 50 كل على حدة
individual_preds = np.array([tree.predict(ambiguous_sample)[0] for tree in forest.estimators_])

print(f"توزيع أصوات الأشجار الخمسين حول العينة: {np.bincount(individual_preds.astype(int), minlength=3)}")
print(f"الانحراف المعياري لأصوات الأشجار: {individual_preds.std():.3f}")
print("دلالة التباين العالي: العينة تقع في منطقة غموض عالية على حدود القرار وتستدعي حذر النموذج!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

# 1. شجرة قرار مفردة (عالية التباين)
tree = DecisionTreeClassifier(random_state=42)
score_tree = cross_val_score(tree, X, y, cv=5).mean()

# 2. غابة عشوائية مكونة من 100 شجرة
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
score_rf = cross_val_score(rf, X, y, cv=5).mean()

print(f"دقة شجرة القرار المفردة: {score_tree:.4f}")
print(f"دقة الغابة العشوائية (100 شجرة): {score_rf:.4f}")
print(f"مكسب التعميم التجميعي: {score_rf - score_tree:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

housing = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    housing.data[:1500], housing.target[:1500], test_size=0.3, random_state=42
)

# اختبار تأثير max_features على خطأ الانحدار
for mf in [1.0, "sqrt", 0.33]:
    rf_reg = RandomForestRegressor(
        n_estimators=60, max_features=mf, random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, rf_reg.predict(X_te))
    print(f"معامل max_features={str(mf):<6} | RMSE الاختبار: {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time

X, y = load_breast_cancer(return_X_y=True)

# مقارنة زمن التدريب والدقة بين RandomForest و ExtraTrees
t0 = time.perf_counter()
rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X, y)
t_rf = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
et = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X, y)
t_et = (time.perf_counter() - t0) * 1000

print(f"RandomForest | الدقة: {rf.score(X, y):.4f} | الزمن: {t_rf:.2f} ms")
print(f"ExtraTrees   | الدقة: {et.score(X, y):.4f} | الزمن: {t_et:.2f} ms")
print(f"ExtraTrees أسرع بنسبة: {(t_rf - t_et) / t_rf:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)

# تفعيل oob_score=True ومراقبة الاستقرار
for n_trees in [10, 30, 70, 150]:
    forest = RandomForestClassifier(
        n_estimators=n_trees, oob_score=True, random_state=42, n_jobs=-1
    ).fit(X, y)
    print(f"أشجار: {n_trees:3d} | دقة التدريب: {forest.score(X, y):.4f} | درجة OOB: {forest.oob_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import time

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

# RandomForest
t0 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rf = time.perf_counter() - t0
rmse_rf = root_mean_squared_error(y_te, rf.predict(X_te))

# ExtraTrees
t0 = time.perf_counter()
et = ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_et = time.perf_counter() - t0
rmse_et = root_mean_squared_error(y_te, et.predict(X_te))

print(f"RandomForest | RMSE: {rmse_rf:.4f} | زمن التدريب: {t_rf * 1000:.1f} ms")
print(f"ExtraTrees   | RMSE: {rmse_et:.4f} | زمن التدريب: {t_et * 1000:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-43-ch43.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_iris(return_X_y=True)
forest = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

# عينة اختبار حدودية بين فئتين
ambiguous_sample = np.array([[6.0, 2.7, 5.1, 1.6]])

# جمع توقعات جميع الأشجار الـ 50 كل على حدة
individual_preds = np.array([tree.predict(ambiguous_sample)[0] for tree in forest.estimators_])

print(f"توزيع أصوات الأشجار الخمسين حول العينة: {np.bincount(individual_preds.astype(int), minlength=3)}")
print(f"الانحراف المعياري لأصوات الأشجار: {individual_preds.std():.3f}")
print("دلالة التباين العالي: العينة تقع في منطقة غموض عالية على حدود القرار وتستدعي حذر النموذج!")
