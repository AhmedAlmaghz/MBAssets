#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — GridSearchCV: البحث المنضبط
Mastering scikit-learn 1.9 — GridSearchCV

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-81-ch81 (lesson 81)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-81-ch81.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — GridSearchCV: البحث المنضبط
Mastering scikit-learn 1.9 — GridSearchCV

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-81-ch81 (lesson 81)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import ParameterGrid

# قاموس واحد: الضرب الديكارتي الكامل
grid_full = {"C": [0.1, 1, 10], "gamma": [0.001, 0.1]}
pg_full = ParameterGrid(grid_full)
print(f"قاموس واحد: {len(grid_full['C'])} × {len(grid_full['gamma'])} = {len(pg_full)} نقطة")

# قائمة قاموسين: شبكتان منفصلتان تُجمَعان
grid_split = [
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
    {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.001, 0.1, 1.0]},
]
pg_split = ParameterGrid(grid_split)
print(f"قاموسان: {3} + ({2} × {3}) = {len(pg_split)} نقطة")

for params in list(pg_split)[:4]:
    print(params)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
param_grid = {"clf__C": [0.1, 1, 10, 100],
              "clf__gamma": [0.0001, 0.001, 0.01, 0.1]}

gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, return_train_score=True)
gs.fit(X_tr, y_tr)

print(f"أفضل تركيبة best_params_: {gs.best_params_}")
print(f"أفضل درجة تحقق best_score_: {gs.best_score_:.4f}")
print(f"دقة best_estimator_ على الاختبار المحجوز: {gs.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

X, y = make_classification(n_samples=160, n_features=200, n_informative=3,
                           n_redundant=2, class_sep=0.8, random_state=42)

# الخطأ: اختيار الخصائص على كل البيانات قبل البحث
X_sel = SelectKBest(f_classif, k=5).fit_transform(X, y)
leaky = cross_val_score(
    GridSearchCV(SVC(kernel="rbf"), {"C": [0.1, 1, 10]}, cv=4, n_jobs=-1),
    X_sel, y, cv=5)

# الصحيح: الاختيار داخل الأنبوب يُعاد ضبطه في كل طية
clean = cross_val_score(
    GridSearchCV(make_pipeline(SelectKBest(f_classif, k=5), SVC(kernel="rbf")),
                 {"svc__C": [0.1, 1, 10]}, cv=4, n_jobs=-1),
    X, y, cv=5)

print(f"التقدير المسرَّب:   {leaky.mean():.4f}")
print(f"التقدير النظيف:    {clean.mean():.4f}")
print(f"الوهم المتفائل:    {leaky.mean() - clean.mean():+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

cvres = pd.DataFrame(gs.cv_results_)
cols = ["param_clf__C", "param_clf__gamma",
        "mean_test_score", "std_test_score", "rank_test_score"]
print(cvres[cols].sort_values("rank_test_score").head(6).to_string(index=False))
print(f"\nbest_index_ = {gs.best_index_} (صف الفائز في الجدول)")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import ParameterGrid

n_combos = len(ParameterGrid(param_grid))
folds = 5

# قياس زمن تدريب واحد للأنبوب على حجم طية تقريبي (٤/٥ بيانات التدريب)
n_fold = int(4 * len(X_tr) / 5)
t0 = time.perf_counter()
pipe.fit(X_tr[:n_fold], y_tr[:n_fold])
t_one = time.perf_counter() - t0

estimate = n_combos * folds * t_one
t0 = time.perf_counter()
gs.fit(X_tr, y_tr)
actual = time.perf_counter() - t0

print(f"التركيبات: {n_combos} × الطيات: {folds} = {n_combos * folds} تدريباً كاملاً")
print(f"زمن تدريب واحد: {t_one * 1000:.1f} ms → تقدير إجمالي: {estimate:.2f} s")
print(f"الزمن الفعلي للبحث: {actual:.2f} s (بالتوازي عبر الأنوية)")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def smart_refit(cv_results):
    """اختر الأقل تشتتاً بين المرشحين القريبين إحصائياً من القمة."""
    scores = cv_results["mean_test_score"]
    stds = cv_results["std_test_score"]
    threshold = scores.max() - stds[scores.argmax()]   # حد القمة ناقص تشتتها
    eligible = scores >= threshold
    return int(np.argmin(np.where(eligible, stds, np.inf)))

gs_smart = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, refit=smart_refit)
gs_smart.fit(X_tr, y_tr)

for name, model in [("الافتراضي", gs), ("الذكي", gs_smart)]:
    idx = model.best_index_
    print(f"{name}: C={gs.cv_results_['param_clf__C'][idx]!s:>5}, "
          f"gamma={gs.cv_results_['param_clf__gamma'][idx]!s:>6} | "
          f"متوسط {gs.cv_results_['mean_test_score'][idx]:.4f}, "
          f"تشتت {gs.cv_results_['std_test_score'][idx]:.4f}, "
          f"اختبار {model.score(X_te, y_te):.4f}")

print(f"هل best_score_ متاح مع refit callable? {hasattr(gs_smart, 'best_score_')}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
gs_err = GridSearchCV(pipe_lr, {"logisticregression__C": [0.01, 1, -5.0, 100]},
                      cv=3, error_score=0.0)
gs_err.fit(X_tr, y_tr)

res = pd.DataFrame(gs_err.cv_results_)
print(res[["param_logisticregression__C", "mean_test_score", "rank_test_score"]]
      .to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

Xw, yw = load_wine(return_X_y=True)
knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_grid = {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11],
            "kneighborsclassifier__weights": ["uniform", "distance"]}
gs_knn = GridSearchCV(knn_pipe, knn_grid, cv=5, n_jobs=-1).fit(Xw, yw)

print(f"أفضل تركيبة: {gs_knn.best_params_}")
print(f"أفضل درجة تحقق: {gs_knn.best_score_:.4f}")
baseline = GridSearchCV(knn_pipe, {}, cv=5, n_jobs=-1).fit(Xw, yw)
print(f"درجة النموذج الافتراضي: {baseline.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part09/ch81.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline

try:
    GridSearchCV(pipe, {"C": [1, 10]}).fit(X_tr, y_tr)
except Exception as err:
    print(f"نوع الخطأ: {type(err).__name__}")
    print(f"الرسالة: {str(err)[:120]}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import ParameterGrid

# قاموس واحد: الضرب الديكارتي الكامل
grid_full = {"C": [0.1, 1, 10], "gamma": [0.001, 0.1]}
pg_full = ParameterGrid(grid_full)
print(f"قاموس واحد: {len(grid_full['C'])} × {len(grid_full['gamma'])} = {len(pg_full)} نقطة")

# قائمة قاموسين: شبكتان منفصلتان تُجمَعان
grid_split = [
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
    {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.001, 0.1, 1.0]},
]
pg_split = ParameterGrid(grid_split)
print(f"قاموسان: {3} + ({2} × {3}) = {len(pg_split)} نقطة")

for params in list(pg_split)[:4]:
    print(params)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
param_grid = {"clf__C": [0.1, 1, 10, 100],
              "clf__gamma": [0.0001, 0.001, 0.01, 0.1]}

gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, return_train_score=True)
gs.fit(X_tr, y_tr)

print(f"أفضل تركيبة best_params_: {gs.best_params_}")
print(f"أفضل درجة تحقق best_score_: {gs.best_score_:.4f}")
print(f"دقة best_estimator_ على الاختبار المحجوز: {gs.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

X, y = make_classification(n_samples=160, n_features=200, n_informative=3,
                           n_redundant=2, class_sep=0.8, random_state=42)

# الخطأ: اختيار الخصائص على كل البيانات قبل البحث
X_sel = SelectKBest(f_classif, k=5).fit_transform(X, y)
leaky = cross_val_score(
    GridSearchCV(SVC(kernel="rbf"), {"C": [0.1, 1, 10]}, cv=4, n_jobs=-1),
    X_sel, y, cv=5)

# الصحيح: الاختيار داخل الأنبوب يُعاد ضبطه في كل طية
clean = cross_val_score(
    GridSearchCV(make_pipeline(SelectKBest(f_classif, k=5), SVC(kernel="rbf")),
                 {"svc__C": [0.1, 1, 10]}, cv=4, n_jobs=-1),
    X, y, cv=5)

print(f"التقدير المسرَّب:   {leaky.mean():.4f}")
print(f"التقدير النظيف:    {clean.mean():.4f}")
print(f"الوهم المتفائل:    {leaky.mean() - clean.mean():+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

cvres = pd.DataFrame(gs.cv_results_)
cols = ["param_clf__C", "param_clf__gamma",
        "mean_test_score", "std_test_score", "rank_test_score"]
print(cvres[cols].sort_values("rank_test_score").head(6).to_string(index=False))
print(f"\nbest_index_ = {gs.best_index_} (صف الفائز في الجدول)")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import ParameterGrid

n_combos = len(ParameterGrid(param_grid))
folds = 5

# قياس زمن تدريب واحد للأنبوب على حجم طية تقريبي (٤/٥ بيانات التدريب)
n_fold = int(4 * len(X_tr) / 5)
t0 = time.perf_counter()
pipe.fit(X_tr[:n_fold], y_tr[:n_fold])
t_one = time.perf_counter() - t0

estimate = n_combos * folds * t_one
t0 = time.perf_counter()
gs.fit(X_tr, y_tr)
actual = time.perf_counter() - t0

print(f"التركيبات: {n_combos} × الطيات: {folds} = {n_combos * folds} تدريباً كاملاً")
print(f"زمن تدريب واحد: {t_one * 1000:.1f} ms → تقدير إجمالي: {estimate:.2f} s")
print(f"الزمن الفعلي للبحث: {actual:.2f} s (بالتوازي عبر الأنوية)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def smart_refit(cv_results):
    """اختر الأقل تشتتاً بين المرشحين القريبين إحصائياً من القمة."""
    scores = cv_results["mean_test_score"]
    stds = cv_results["std_test_score"]
    threshold = scores.max() - stds[scores.argmax()]   # حد القمة ناقص تشتتها
    eligible = scores >= threshold
    return int(np.argmin(np.where(eligible, stds, np.inf)))

gs_smart = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, refit=smart_refit)
gs_smart.fit(X_tr, y_tr)

for name, model in [("الافتراضي", gs), ("الذكي", gs_smart)]:
    idx = model.best_index_
    print(f"{name}: C={gs.cv_results_['param_clf__C'][idx]!s:>5}, "
          f"gamma={gs.cv_results_['param_clf__gamma'][idx]!s:>6} | "
          f"متوسط {gs.cv_results_['mean_test_score'][idx]:.4f}, "
          f"تشتت {gs.cv_results_['std_test_score'][idx]:.4f}, "
          f"اختبار {model.score(X_te, y_te):.4f}")

print(f"هل best_score_ متاح مع refit callable? {hasattr(gs_smart, 'best_score_')}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
gs_err = GridSearchCV(pipe_lr, {"logisticregression__C": [0.01, 1, -5.0, 100]},
                      cv=3, error_score=0.0)
gs_err.fit(X_tr, y_tr)

res = pd.DataFrame(gs_err.cv_results_)
print(res[["param_logisticregression__C", "mean_test_score", "rank_test_score"]]
      .to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

Xw, yw = load_wine(return_X_y=True)
knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_grid = {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11],
            "kneighborsclassifier__weights": ["uniform", "distance"]}
gs_knn = GridSearchCV(knn_pipe, knn_grid, cv=5, n_jobs=-1).fit(Xw, yw)

print(f"أفضل تركيبة: {gs_knn.best_params_}")
print(f"أفضل درجة تحقق: {gs_knn.best_score_:.4f}")
baseline = GridSearchCV(knn_pipe, {}, cv=5, n_jobs=-1).fit(Xw, yw)
print(f"درجة النموذج الافتراضي: {baseline.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-81-ch81.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline

try:
    GridSearchCV(pipe, {"C": [1, 10]}).fit(X_tr, y_tr)
except Exception as err:
    print(f"نوع الخطأ: {type(err).__name__}")
    print(f"الرسالة: {str(err)[:120]}")
