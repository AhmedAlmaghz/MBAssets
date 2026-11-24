#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط شجرة القرار
Mastering scikit-learn 1.9 — Chapter 40 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-40-ch40 (lesson 40)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-40-ch40.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط شجرة القرار
Mastering scikit-learn 1.9 — Chapter 40 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-40-ch40 (lesson 40)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# مقارنة بين تقييد العمق max_depth وتقييد الأوراق max_leaf_nodes
tree_depth = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
tree_leaves = DecisionTreeClassifier(max_leaf_nodes=8, random_state=42).fit(X_train, y_train)

print(f"تقييد max_depth=4    | عدد الأوراق: {tree_depth.get_n_leaves():2d} | دقة الاختبار: {tree_depth.score(X_test, y_test):.4f}")
print(f"تقييد max_leaf_nodes=8 | عدد الأوراق: {tree_leaves.get_n_leaves():2d} | دقة الاختبار: {tree_leaves.score(X_test, y_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
# فئة نادرة بنسبة 5% فقط
y = rng.choice([0, 1], size=500, p=[0.95, 0.05])

clf_default = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
clf_balanced = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X, y)

pred_def = clf_default.predict(X)
pred_bal = clf_balanced.predict(X)

print(f"عدد العينات المتنبأ بها كفئة نادرة (الافتراضي): {np.sum(pred_def == 1)} (تجاهل شبه تام)")
print(f"عدد العينات المتنبأ بها كفئة نادرة (balanced):   {np.sum(pred_bal == 1)} (حساسية واستجابة حقيقية)")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# مصفوفة بيانات تحتوي قيماً مفقودة صريحة
X_missing = np.array([
    [2.5, np.nan],
    [np.nan, 1.2],
    [3.1, 0.8],
    [1.1, np.nan],
    [np.nan, 3.5],
    [4.0, 2.9]
])
y_missing = np.array([0, 1, 0, 0, 1, 1])

# التدريب مباشرة دون أي Imputer خارجي!
tree_nan = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_nan.fit(X_missing, y_missing)

# التنبؤ لعينة جديدة تحوي قيماً مفقودة
X_new = np.array([[np.nan, 0.5], [3.5, np.nan]])
predictions = tree_nan.predict(X_new)

print("تم تدريب الشجرة على بيانات تحوي np.nan مباشرة بنجاح!")
print(f"التنبؤ للعينات الجديدة المحتوية على قيم مفقودة: {list(predictions)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

X, y = load_breast_cancer(return_X_y=True)

# فضاء المعلمات المرجعي لشجرة متزنة
param_dist = {
    "max_depth": randint(2, 10),
    "min_samples_leaf": randint(2, 20),
    "max_leaf_nodes": randint(6, 30),
    "criterion": ["gini", "entropy", "log_loss"]
}

search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
search.fit(X, y)

print(f"أفضل نتيجة ROC-AUC: {search.best_score_:.4f}")
print("المعلمات الفائقة المثلى:")
for param, val in search.best_params_.items():
    print(f"  {param:<18}: {val}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1000], data.target[:1000], test_size=0.3, random_state=42
)

t1 = DecisionTreeRegressor(min_samples_leaf=1, random_state=42).fit(X_tr, y_tr)
t20 = DecisionTreeRegressor(min_samples_leaf=20, random_state=42).fit(X_tr, y_tr)

print(f"min_samples_leaf=1  | أوراق: {t1.get_n_leaves():3d} | MSE تدريب: {mean_squared_error(y_tr, t1.predict(X_tr)):.4f} | MSE اختبار: {mean_squared_error(y_te, t1.predict(X_te)):.4f}")
print(f"min_samples_leaf=20 | أوراق: {t20.get_n_leaves():3d} | MSE تدريب: {mean_squared_error(y_tr, t20.predict(X_tr)):.4f} | MSE اختبار: {mean_squared_error(y_te, t20.predict(X_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part05/ch40.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(42)

# إتلاف 20% من القيم بجعلها مفقودة
X_corrupt = X.copy()
mask = rng.uniform(0, 1, size=X.shape) < 0.2
X_corrupt[mask] = np.nan

# 1. الدعم الأصيل في scikit-learn
score_native = cross_val_score(DecisionTreeClassifier(random_state=42), X_corrupt, y, cv=5).mean()

# 2. التعويض بالمتوسط عبر SimpleImputer
pipe_impute = make_pipeline(SimpleImputer(strategy="mean"), DecisionTreeClassifier(random_state=42))
score_impute = cross_val_score(pipe_impute, X_corrupt, y, cv=5).mean()

print(f"دقة الدعم الأصيل المباشر (Native Missing): {score_native:.4f}")
print(f"دقة التعويض التقليدي (SimpleImputer):      {score_impute:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# مقارنة بين تقييد العمق max_depth وتقييد الأوراق max_leaf_nodes
tree_depth = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
tree_leaves = DecisionTreeClassifier(max_leaf_nodes=8, random_state=42).fit(X_train, y_train)

print(f"تقييد max_depth=4    | عدد الأوراق: {tree_depth.get_n_leaves():2d} | دقة الاختبار: {tree_depth.score(X_test, y_test):.4f}")
print(f"تقييد max_leaf_nodes=8 | عدد الأوراق: {tree_leaves.get_n_leaves():2d} | دقة الاختبار: {tree_leaves.score(X_test, y_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
# فئة نادرة بنسبة 5% فقط
y = rng.choice([0, 1], size=500, p=[0.95, 0.05])

clf_default = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
clf_balanced = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X, y)

pred_def = clf_default.predict(X)
pred_bal = clf_balanced.predict(X)

print(f"عدد العينات المتنبأ بها كفئة نادرة (الافتراضي): {np.sum(pred_def == 1)} (تجاهل شبه تام)")
print(f"عدد العينات المتنبأ بها كفئة نادرة (balanced):   {np.sum(pred_bal == 1)} (حساسية واستجابة حقيقية)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# مصفوفة بيانات تحتوي قيماً مفقودة صريحة
X_missing = np.array([
    [2.5, np.nan],
    [np.nan, 1.2],
    [3.1, 0.8],
    [1.1, np.nan],
    [np.nan, 3.5],
    [4.0, 2.9]
])
y_missing = np.array([0, 1, 0, 0, 1, 1])

# التدريب مباشرة دون أي Imputer خارجي!
tree_nan = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_nan.fit(X_missing, y_missing)

# التنبؤ لعينة جديدة تحوي قيماً مفقودة
X_new = np.array([[np.nan, 0.5], [3.5, np.nan]])
predictions = tree_nan.predict(X_new)

print("تم تدريب الشجرة على بيانات تحوي np.nan مباشرة بنجاح!")
print(f"التنبؤ للعينات الجديدة المحتوية على قيم مفقودة: {list(predictions)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

X, y = load_breast_cancer(return_X_y=True)

# فضاء المعلمات المرجعي لشجرة متزنة
param_dist = {
    "max_depth": randint(2, 10),
    "min_samples_leaf": randint(2, 20),
    "max_leaf_nodes": randint(6, 30),
    "criterion": ["gini", "entropy", "log_loss"]
}

search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
search.fit(X, y)

print(f"أفضل نتيجة ROC-AUC: {search.best_score_:.4f}")
print("المعلمات الفائقة المثلى:")
for param, val in search.best_params_.items():
    print(f"  {param:<18}: {val}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1000], data.target[:1000], test_size=0.3, random_state=42
)

t1 = DecisionTreeRegressor(min_samples_leaf=1, random_state=42).fit(X_tr, y_tr)
t20 = DecisionTreeRegressor(min_samples_leaf=20, random_state=42).fit(X_tr, y_tr)

print(f"min_samples_leaf=1  | أوراق: {t1.get_n_leaves():3d} | MSE تدريب: {mean_squared_error(y_tr, t1.predict(X_tr)):.4f} | MSE اختبار: {mean_squared_error(y_te, t1.predict(X_te)):.4f}")
print(f"min_samples_leaf=20 | أوراق: {t20.get_n_leaves():3d} | MSE تدريب: {mean_squared_error(y_tr, t20.predict(X_tr)):.4f} | MSE اختبار: {mean_squared_error(y_te, t20.predict(X_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-40-ch40.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(42)

# إتلاف 20% من القيم بجعلها مفقودة
X_corrupt = X.copy()
mask = rng.uniform(0, 1, size=X.shape) < 0.2
X_corrupt[mask] = np.nan

# 1. الدعم الأصيل في scikit-learn
score_native = cross_val_score(DecisionTreeClassifier(random_state=42), X_corrupt, y, cv=5).mean()

# 2. التعويض بالمتوسط عبر SimpleImputer
pipe_impute = make_pipeline(SimpleImputer(strategy="mean"), DecisionTreeClassifier(random_state=42))
score_impute = cross_val_score(pipe_impute, X_corrupt, y, cv=5).mean()

print(f"دقة الدعم الأصيل المباشر (Native Missing): {score_native:.4f}")
print(f"دقة التعويض التقليدي (SimpleImputer):      {score_impute:.4f}")
