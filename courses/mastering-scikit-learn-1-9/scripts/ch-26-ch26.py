#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المعلومات المتبادلة والاختيار غير المشرف
Mastering scikit-learn 1.9 — Chapter 26 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-26-ch26 (lesson 26)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-26-ch26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المعلومات المتبادلة والاختيار غير المشرف
Mastering scikit-learn 1.9 — Chapter 26 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-26-ch26 (lesson 26)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import numpy as np

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

mi_scores = mutual_info_classif(X, y, random_state=42)

print("درجات المعلومات المتبادلة (Iris):")
for name, score in sorted(zip(feature_names, mi_scores),
                           key=lambda t: -t[1]):
    bar = "█" * int(score * 20)
    print(f"  {name:<25} {score:.4f}  {bar}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

rng = np.random.default_rng(42)
n = 500

# ميزة 1: علاقة خطية قوية مع الهدف
x_linear = rng.standard_normal(n)
y = (x_linear > 0).astype(int)

# ميزة 2: علاقة غير خطية (تربيعية) — لا ارتباط خطي
x_nonlinear = rng.standard_normal(n)
# الهدف يعتمد على قيمة x² ولكن لا يوجد ارتباط خطي
y_nl = (x_nonlinear**2 > 0.5).astype(int)

X_test = np.column_stack([x_linear, x_nonlinear])

f_scores, _ = f_classif(X_test, y)
mi_scores = mutual_info_classif(X_test, y, random_state=42)

print("مقارنة F-test و MI:")
print(f"{'الميزة':<20} {'F-score':>10} {'MI-score':>10}")
print("-" * 42)
print(f"{'خطية':<20} {f_scores[0]:>10.3f} {mi_scores[0]:>10.4f}")
print(f"{'غير خطية':<20} {f_scores[1]:>10.3f} {mi_scores[1]:>10.4f}")
print()
print("→ F-test يُفضّل الخطية، MI يكتشف الغير خطية أيضاً")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

mi_scores = mutual_info_regression(X, y, random_state=42)
indices = np.argsort(mi_scores)[::-1]

print("المعلومات المتبادلة مع أسعار المنازل (California Housing):")
for idx in indices:
    bar = "█" * int(mi_scores[idx] * 10)
    print(f"  {feature_names[idx]:<12} {mi_scores[idx]:.4f}  {bar}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((300, 50))

# أضف ميزات شبه ثابتة
X[:, 0] = 0.0          # ثابتة تماماً
X[:, 1] = rng.choice([0, 1], size=300, p=[0.99, 0.01])  # شبه ثابتة

selector = VarianceThreshold(threshold=0.01)
X_sel = selector.fit_transform(X)
print(f"الميزات الأصلية: {X.shape[1]}")
print(f"الميزات بعد VarianceThreshold(0.01): {X_sel.shape[1]}")
print(f"الميزات المحذوفة: {X.shape[1] - X_sel.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = load_digits(return_X_y=True)
print(f"الأبعاد الأصلية: {X.shape}")

# نُقلِّص إلى الأبعاد التي تشرح 95% من التباين
pca = PCA(n_components=0.95, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)

print(f"الأبعاد بعد PCA (95% variance): {X_pca.shape}")
print(f"نسبة الاختزال: {1 - X_pca.shape[1]/X.shape[1]:.1%}")
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"أول مكوّن يشرح: {pca.explained_variance_ratio_[0]:.2%}")
print(f"أول 10 مكونات تشرح: {cumvar[9]:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)

f_scores, _ = f_classif(X, y)
mi_scores = mutual_info_classif(X, y, random_state=42)

top10_f = set(np.argsort(f_scores)[::-1][:10])
top10_mi = set(np.argsort(mi_scores)[::-1][:10])
overlap = top10_f & top10_mi

print(f"ميزات f_classif: {[feature_names[i][:12] for i in top10_f]}")
print(f"ميزات MI:        {[feature_names[i][:12] for i in top10_mi]}")
print(f"مشتركة: {len(overlap)}/10")

pipe_f = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("clf", lr)])
pipe_mi = Pipeline([("sel", SelectKBest(mutual_info_classif, k=10)), ("clf", lr)])

score_f = cross_val_score(pipe_f, X, y, cv=cv).mean()
score_mi = cross_val_score(pipe_mi, X, y, cv=cv).mean()
print(f"\nدقة f_classif pipeline:  {score_f:.4f}")
print(f"دقة MI pipeline:          {score_mi:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part03/ch26.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_digits(return_X_y=True)
sel = VarianceThreshold(threshold=0)
X_sel = sel.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
score_all = cross_val_score(knn, X, y, cv=5).mean()
score_sel = cross_val_score(knn, X_sel, y, cv=5).mean()

print(f"الأبعاد الأصلية: {X.shape[1]}  → دقة: {score_all:.4f}")
print(f"الأبعاد بعد حذف الثابتة: {X_sel.shape[1]}  → دقة: {score_sel:.4f}")
print(f"البكسلات المحذوفة: {X.shape[1] - X_sel.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import numpy as np

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

mi_scores = mutual_info_classif(X, y, random_state=42)

print("درجات المعلومات المتبادلة (Iris):")
for name, score in sorted(zip(feature_names, mi_scores),
                           key=lambda t: -t[1]):
    bar = "█" * int(score * 20)
    print(f"  {name:<25} {score:.4f}  {bar}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

rng = np.random.default_rng(42)
n = 500

# ميزة 1: علاقة خطية قوية مع الهدف
x_linear = rng.standard_normal(n)
y = (x_linear > 0).astype(int)

# ميزة 2: علاقة غير خطية (تربيعية) — لا ارتباط خطي
x_nonlinear = rng.standard_normal(n)
# الهدف يعتمد على قيمة x² ولكن لا يوجد ارتباط خطي
y_nl = (x_nonlinear**2 > 0.5).astype(int)

X_test = np.column_stack([x_linear, x_nonlinear])

f_scores, _ = f_classif(X_test, y)
mi_scores = mutual_info_classif(X_test, y, random_state=42)

print("مقارنة F-test و MI:")
print(f"{'الميزة':<20} {'F-score':>10} {'MI-score':>10}")
print("-" * 42)
print(f"{'خطية':<20} {f_scores[0]:>10.3f} {mi_scores[0]:>10.4f}")
print(f"{'غير خطية':<20} {f_scores[1]:>10.3f} {mi_scores[1]:>10.4f}")
print()
print("→ F-test يُفضّل الخطية، MI يكتشف الغير خطية أيضاً")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

mi_scores = mutual_info_regression(X, y, random_state=42)
indices = np.argsort(mi_scores)[::-1]

print("المعلومات المتبادلة مع أسعار المنازل (California Housing):")
for idx in indices:
    bar = "█" * int(mi_scores[idx] * 10)
    print(f"  {feature_names[idx]:<12} {mi_scores[idx]:.4f}  {bar}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((300, 50))

# أضف ميزات شبه ثابتة
X[:, 0] = 0.0          # ثابتة تماماً
X[:, 1] = rng.choice([0, 1], size=300, p=[0.99, 0.01])  # شبه ثابتة

selector = VarianceThreshold(threshold=0.01)
X_sel = selector.fit_transform(X)
print(f"الميزات الأصلية: {X.shape[1]}")
print(f"الميزات بعد VarianceThreshold(0.01): {X_sel.shape[1]}")
print(f"الميزات المحذوفة: {X.shape[1] - X_sel.shape[1]}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = load_digits(return_X_y=True)
print(f"الأبعاد الأصلية: {X.shape}")

# نُقلِّص إلى الأبعاد التي تشرح 95% من التباين
pca = PCA(n_components=0.95, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)

print(f"الأبعاد بعد PCA (95% variance): {X_pca.shape}")
print(f"نسبة الاختزال: {1 - X_pca.shape[1]/X.shape[1]:.1%}")
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"أول مكوّن يشرح: {pca.explained_variance_ratio_[0]:.2%}")
print(f"أول 10 مكونات تشرح: {cumvar[9]:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)

f_scores, _ = f_classif(X, y)
mi_scores = mutual_info_classif(X, y, random_state=42)

top10_f = set(np.argsort(f_scores)[::-1][:10])
top10_mi = set(np.argsort(mi_scores)[::-1][:10])
overlap = top10_f & top10_mi

print(f"ميزات f_classif: {[feature_names[i][:12] for i in top10_f]}")
print(f"ميزات MI:        {[feature_names[i][:12] for i in top10_mi]}")
print(f"مشتركة: {len(overlap)}/10")

pipe_f = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("clf", lr)])
pipe_mi = Pipeline([("sel", SelectKBest(mutual_info_classif, k=10)), ("clf", lr)])

score_f = cross_val_score(pipe_f, X, y, cv=cv).mean()
score_mi = cross_val_score(pipe_mi, X, y, cv=cv).mean()
print(f"\nدقة f_classif pipeline:  {score_f:.4f}")
print(f"دقة MI pipeline:          {score_mi:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-26-ch26.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_digits(return_X_y=True)
sel = VarianceThreshold(threshold=0)
X_sel = sel.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
score_all = cross_val_score(knn, X, y, cv=5).mean()
score_sel = cross_val_score(knn, X_sel, y, cv=5).mean()

print(f"الأبعاد الأصلية: {X.shape[1]}  → دقة: {score_all:.4f}")
print(f"الأبعاد بعد حذف الثابتة: {X_sel.shape[1]}  → دقة: {score_sel:.4f}")
print(f"البكسلات المحذوفة: {X.shape[1] - X_sel.shape[1]}")
