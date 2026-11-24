#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الاختيار داخل خط الأنابيب
Mastering scikit-learn 1.9 — Chapter 24 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-24-ch24 (lesson 24)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-24-ch24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الاختيار داخل خط الأنابيب
Mastering scikit-learn 1.9 — Chapter 24 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-24-ch24 (lesson 24)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# نبني مجموعة بيانات صعبة: 500 ميزة، 20 فقط مفيدة فعلاً
X, y = make_classification(
    n_samples=200,
    n_features=500,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)

# ===== الطريقة الخاطئة: نختار الميزات على كامل X أولاً =====
selector = SelectKBest(f_classif, k=20)
selector.fit(X, y)          # <-- هنا يُشاهَد الاختبار كله!
X_selected = selector.transform(X)

scores_leaky = cross_val_score(lr, X_selected, y, cv=cv, scoring="accuracy")
print(f"[تسرّب]  دقة CV: {scores_leaky.mean():.4f} ± {scores_leaky.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = make_classification(
    n_samples=200,
    n_features=500,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===== الطريقة الصحيحة: Pipeline يُعزل الاختيار داخل كل طية =====
pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=20)),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
])

scores_clean = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"[صحيح]   دقة CV: {scores_clean.mean():.4f} ± {scores_clean.std():.4f}")
print()
print("الفارق:", round(0.98 - scores_clean.mean(), 4),
      "← هذا ما يصنعه التسرّب!")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline مع SelectFromModel + LogisticRegression
pipe = Pipeline([
    ("selector", SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42),
        threshold="mean"
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"دقة Pipeline مع SelectFromModel: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"الميزات الأصلية: {X.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_rfe = Pipeline([
    ("rfe", RFE(
        estimator=LogisticRegression(C=0.5, max_iter=3000, random_state=42),
        n_features_to_select=10,
        step=1
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

scores_rfe = cross_val_score(pipe_rfe, X, y, cv=cv, scoring="accuracy")
print(f"دقة Pipeline مع RFE(10): {scores_rfe.mean():.4f} ± {scores_rfe.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(f_classif)),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

# نبحث عن أفضل k من {5, 10, 15, 20}
param_grid = {"selector__k": [5, 10, 15, 20]}
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)

print(f"أفضل k: {grid.best_params_['selector__k']}")
print(f"أفضل دقة CV: {grid.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=10)),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe.fit(X, y)

# نستخرج الميزات المختارة من داخل الأنبوب المُدرَّب
selector = pipe.named_steps["selector"]
chosen = feature_names[selector.get_support()]
print(f"أفضل 10 ميزات (من داخل Pipeline):")
for name in chosen:
    print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=300, n_features=200,
                            n_informative=10, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=3000, random_state=42)

# تسرّب
sel = SelectKBest(f_classif, k=10)
X_sel = sel.fit_transform(X, y)
score_leak = cross_val_score(lr, X_sel, y, cv=cv).mean()

# صحيح
pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("clf", lr)])
score_clean = cross_val_score(pipe, X, y, cv=cv).mean()

print(f"دقة مع تسرّب:  {score_leak:.4f}")
print(f"دقة صحيحة:     {score_clean:.4f}")
print(f"فارق التسرّب:  {score_leak - score_clean:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part03/ch24.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)

pipe = Pipeline([
    ("sel", SelectKBest(f_classif)),
    ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=42))
])

grid = GridSearchCV(pipe, {"sel__k": [5, 10, 15, 20, 25]},
                    cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)
print(f"أفضل k: {grid.best_params_['sel__k']}")
print(f"أفضل دقة: {grid.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# نبني مجموعة بيانات صعبة: 500 ميزة، 20 فقط مفيدة فعلاً
X, y = make_classification(
    n_samples=200,
    n_features=500,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)

# ===== الطريقة الخاطئة: نختار الميزات على كامل X أولاً =====
selector = SelectKBest(f_classif, k=20)
selector.fit(X, y)          # <-- هنا يُشاهَد الاختبار كله!
X_selected = selector.transform(X)

scores_leaky = cross_val_score(lr, X_selected, y, cv=cv, scoring="accuracy")
print(f"[تسرّب]  دقة CV: {scores_leaky.mean():.4f} ± {scores_leaky.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = make_classification(
    n_samples=200,
    n_features=500,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===== الطريقة الصحيحة: Pipeline يُعزل الاختيار داخل كل طية =====
pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=20)),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
])

scores_clean = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"[صحيح]   دقة CV: {scores_clean.mean():.4f} ± {scores_clean.std():.4f}")
print()
print("الفارق:", round(0.98 - scores_clean.mean(), 4),
      "← هذا ما يصنعه التسرّب!")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline مع SelectFromModel + LogisticRegression
pipe = Pipeline([
    ("selector", SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42),
        threshold="mean"
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"دقة Pipeline مع SelectFromModel: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"الميزات الأصلية: {X.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_rfe = Pipeline([
    ("rfe", RFE(
        estimator=LogisticRegression(C=0.5, max_iter=3000, random_state=42),
        n_features_to_select=10,
        step=1
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

scores_rfe = cross_val_score(pipe_rfe, X, y, cv=cv, scoring="accuracy")
print(f"دقة Pipeline مع RFE(10): {scores_rfe.mean():.4f} ± {scores_rfe.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(f_classif)),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])

# نبحث عن أفضل k من {5, 10, 15, 20}
param_grid = {"selector__k": [5, 10, 15, 20]}
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)

print(f"أفضل k: {grid.best_params_['selector__k']}")
print(f"أفضل دقة CV: {grid.best_score_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=10)),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe.fit(X, y)

# نستخرج الميزات المختارة من داخل الأنبوب المُدرَّب
selector = pipe.named_steps["selector"]
chosen = feature_names[selector.get_support()]
print(f"أفضل 10 ميزات (من داخل Pipeline):")
for name in chosen:
    print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=300, n_features=200,
                            n_informative=10, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=3000, random_state=42)

# تسرّب
sel = SelectKBest(f_classif, k=10)
X_sel = sel.fit_transform(X, y)
score_leak = cross_val_score(lr, X_sel, y, cv=cv).mean()

# صحيح
pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("clf", lr)])
score_clean = cross_val_score(pipe, X, y, cv=cv).mean()

print(f"دقة مع تسرّب:  {score_leak:.4f}")
print(f"دقة صحيحة:     {score_clean:.4f}")
print(f"فارق التسرّب:  {score_leak - score_clean:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-24-ch24.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)

pipe = Pipeline([
    ("sel", SelectKBest(f_classif)),
    ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=42))
])

grid = GridSearchCV(pipe, {"sel__k": [5, 10, 15, 20, 25]},
                    cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)
print(f"أفضل k: {grid.best_params_['sel__k']}")
print(f"أفضل دقة: {grid.best_score_:.4f}")
