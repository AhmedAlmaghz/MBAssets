#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التجميع بالعيّنات: Bagging
Mastering scikit-learn 1.9 — Bagging

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-42-ch42 (lesson 42)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-42-ch42.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التجميع بالعيّنات: Bagging
Mastering scikit-learn 1.9 — Bagging

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-42-ch42 (lesson 42)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# محاكاة تجريبية للنسبة المئوية لعينات OOB
n_samples = 10000
rng = np.random.default_rng(42)

indices = np.arange(n_samples)
bootstrap_sample = rng.choice(indices, size=n_samples, replace=True)
unique_selected = len(np.unique(bootstrap_sample))
oob_count = n_samples - unique_selected

print(f"إجمالي حجم البيانات الأصلية: {n_samples}")
print(f"العينات الفريدة المختارة داخل الحقيبة: {unique_selected} ({unique_selected/n_samples:.1%})")
print(f"العينات المتروكة خارج الحقيبة (OOB): {oob_count} ({oob_count/n_samples:.1%})")
print(f"القيمة النظرية 1/e: {1/np.e:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# تفعيل oob_score=True
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)

print(f"درجة التقييم المجانية خارج العينة (OOB Score): {bagging.oob_score_:.4f}")
print(f"دقة الاختبار المستقل الحقيقية (Test Score):   {bagging.score(X_test, y_test):.4f}")
print(f"الفارق بين التقديرين: {abs(bagging.oob_score_ - bagging.score(X_test, y_test)):.4f} (تطابق استثنائي!)")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

# أسلوب Random Patches: عينات 70% من الصفوف و 60% من الميزات
patch_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.7,
    max_features=0.6,
    bootstrap=True,
    bootstrap_features=True,
    random_state=42,
    n_jobs=-1
)
patch_bagging.fit(X, y)
print(f"دقة نموذج Random Patches: {patch_bagging.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)

# مقارنة نموذج الجار الأقرب المفرد مع تجميعه عبر Bagging
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
pipe_bagged_knn = make_pipeline(
    StandardScaler(),
    BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=3),
        n_estimators=30,
        random_state=42
    )
)

score_single = cross_val_score(pipe_knn, X, y, cv=5).mean()
score_bagged = cross_val_score(pipe_bagged_knn, X, y, cv=5).mean()

print(f"متوسط دقة KNN منفرد:  {score_single:.4f}")
print(f"متوسط دقة Bagged KNN: {score_bagged:.4f}")
print(f"التحسن الناتج عن التجميع: {score_bagged - score_single:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

# 1. شجرة انحدار منفردة
single_tree = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
rmse_single = root_mean_squared_error(y_te, single_tree.predict(X_te))

# 2. تجميع BaggingRegressor
bag_tree = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    oob_score=True,
    random_state=42,
    n_jobs=-1
).fit(X_tr, y_tr)
rmse_bag = root_mean_squared_error(y_te, bag_tree.predict(X_te))

print(f"RMSE شجرة انحدار منفردة: {rmse_single:.4f}")
print(f"RMSE تجميع Bagging:       {rmse_bag:.4f}")
print(f"درجة التقييم المجاني OOB R²: {bag_tree.oob_score_:.4f}")
print(f"انخفاض الخطأ بفضل التجميع: {rmse_single - rmse_bag:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch42.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import BaggingClassifier
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 1, 0, 1])

try:
    BaggingClassifier(bootstrap=False, oob_score=True).fit(X, y)
except ValueError as err:
    print(f"خطأ ملتقط بنجاح: {err}")
    print("السبب: إذا كان السحب بدون إرجاع (Pasting)، فإن كل النماذج ترى جميع العينات ولا توجد أي عينة متروكة خارج الحقيبة!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# محاكاة تجريبية للنسبة المئوية لعينات OOB
n_samples = 10000
rng = np.random.default_rng(42)

indices = np.arange(n_samples)
bootstrap_sample = rng.choice(indices, size=n_samples, replace=True)
unique_selected = len(np.unique(bootstrap_sample))
oob_count = n_samples - unique_selected

print(f"إجمالي حجم البيانات الأصلية: {n_samples}")
print(f"العينات الفريدة المختارة داخل الحقيبة: {unique_selected} ({unique_selected/n_samples:.1%})")
print(f"العينات المتروكة خارج الحقيبة (OOB): {oob_count} ({oob_count/n_samples:.1%})")
print(f"القيمة النظرية 1/e: {1/np.e:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# تفعيل oob_score=True
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)

print(f"درجة التقييم المجانية خارج العينة (OOB Score): {bagging.oob_score_:.4f}")
print(f"دقة الاختبار المستقل الحقيقية (Test Score):   {bagging.score(X_test, y_test):.4f}")
print(f"الفارق بين التقديرين: {abs(bagging.oob_score_ - bagging.score(X_test, y_test)):.4f} (تطابق استثنائي!)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

# أسلوب Random Patches: عينات 70% من الصفوف و 60% من الميزات
patch_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.7,
    max_features=0.6,
    bootstrap=True,
    bootstrap_features=True,
    random_state=42,
    n_jobs=-1
)
patch_bagging.fit(X, y)
print(f"دقة نموذج Random Patches: {patch_bagging.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)

# مقارنة نموذج الجار الأقرب المفرد مع تجميعه عبر Bagging
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
pipe_bagged_knn = make_pipeline(
    StandardScaler(),
    BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=3),
        n_estimators=30,
        random_state=42
    )
)

score_single = cross_val_score(pipe_knn, X, y, cv=5).mean()
score_bagged = cross_val_score(pipe_bagged_knn, X, y, cv=5).mean()

print(f"متوسط دقة KNN منفرد:  {score_single:.4f}")
print(f"متوسط دقة Bagged KNN: {score_bagged:.4f}")
print(f"التحسن الناتج عن التجميع: {score_bagged - score_single:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

# 1. شجرة انحدار منفردة
single_tree = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
rmse_single = root_mean_squared_error(y_te, single_tree.predict(X_te))

# 2. تجميع BaggingRegressor
bag_tree = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    oob_score=True,
    random_state=42,
    n_jobs=-1
).fit(X_tr, y_tr)
rmse_bag = root_mean_squared_error(y_te, bag_tree.predict(X_te))

print(f"RMSE شجرة انحدار منفردة: {rmse_single:.4f}")
print(f"RMSE تجميع Bagging:       {rmse_bag:.4f}")
print(f"درجة التقييم المجاني OOB R²: {bag_tree.oob_score_:.4f}")
print(f"انخفاض الخطأ بفضل التجميع: {rmse_single - rmse_bag:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-42-ch42.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import BaggingClassifier
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 1, 0, 1])

try:
    BaggingClassifier(bootstrap=False, oob_score=True).fit(X, y)
except ValueError as err:
    print(f"خطأ ملتقط بنجاح: {err}")
    print("السبب: إذا كان السحب بدون إرجاع (Pasting)، فإن كل النماذج ترى جميع العينات ولا توجد أي عينة متروكة خارج الحقيبة!")
