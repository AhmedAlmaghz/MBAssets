#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الجيران الأقرب: k-NN
Mastering scikit-learn 1.9 — k-NN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-54-ch54 (lesson 54)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-54-ch54.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الجيران الأقرب: k-NN
Mastering scikit-learn 1.9 — k-NN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-54-ch54 (lesson 54)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=40000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for n in [5000, 40000]:
    knn = KNeighborsClassifier(n_neighbors=5)
    t0 = time.perf_counter(); knn.fit(X_tr[:n], y_tr[:n]); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); knn.predict(X_te[:2000]); t_pred = time.perf_counter() - t0
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    t0 = time.perf_counter(); rf.fit(X_tr[:n], y_tr[:n]); t_rf = time.perf_counter() - t0
    print(f"n = {n:6d} | fit k-NN: {t_fit * 1000:6.1f} ms | تنبؤ 2000 استعلام: {t_pred * 1000:7.1f} ms | fit غابة 50 شجرة: {t_rf * 1000:8.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

d = np.array([0.0, 2.5, 7.1])          # جار ملاصق تماماً (d = 0)
with np.errstate(divide="ignore"):
    print(f"أوزان 1/d الساذجة: {1.0 / d}")

Xs = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [5.1, 5.2]])
ys = np.array([0, 0, 1, 1])
knn = KNeighborsClassifier(n_neighbors=4, weights="distance").fit(Xs, ys)
print(f"استعلام مطابق لعينة تدريب: {knn.predict_proba(np.array([[0.0, 0.0]]))}")
print(f"استعلام بعيد عن الجميع:     {knn.predict_proba(np.array([[2.4, 2.4]]))}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(n_samples=800, n_features=14, n_informative=5, random_state=3)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"{'k':>4} | {'دقة التدريب':>10} | {'دقة الاختبار':>11}")
for k in [1, 3, 5, 9, 15, 25, 40]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_tr, y_tr)
    print(f"{k:4d} | {pipe.score(X_tr, y_tr):10.4f} | {pipe.score(X_te, y_te):11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)
stds = X.std(axis=0)
share = (stds ** 2) / (stds ** 2).sum()
print(f"عمود proline وحده يحتكر {share[-1]:.1%} من تباين المسافة الإقليدية قبل التطبيع")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
print(f"دقة بدون تطبيع:  {knn_raw.score(X_te, y_te):.4f}")

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
pipe.fit(X_tr, y_tr)
print(f"دقة مع StandardScaler: {pipe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"{'k':>4} | {'uniform':>8} | {'distance':>8}")
for k in range(1, 22, 2):
    accs = []
    for w in ["uniform", "distance"]:
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w))
        pipe.fit(X_tr, y_tr)
        accs.append(pipe.score(X_te, y_te))
    print(f"{k:4d} | {accs[0]:8.4f} | {accs[1]:8.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch54.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

X, y = make_regression(n_samples=600, n_features=4, noise=25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

cv_scores = {}
for k in range(1, 31, 2):
    p = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    cv_scores[k] = cross_val_score(p, X_tr, y_tr, cv=5,
                                   scoring="neg_root_mean_squared_error").mean()

best_k = max(cv_scores, key=cv_scores.get)
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k))
pipe.fit(X_tr, y_tr)
rmse = root_mean_squared_error(y_te, pipe.predict(X_te))
print(f"أفضل k بالتحقق المتقاطع: {best_k} | RMSE الاختبار: {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=40000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for n in [5000, 40000]:
    knn = KNeighborsClassifier(n_neighbors=5)
    t0 = time.perf_counter(); knn.fit(X_tr[:n], y_tr[:n]); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); knn.predict(X_te[:2000]); t_pred = time.perf_counter() - t0
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    t0 = time.perf_counter(); rf.fit(X_tr[:n], y_tr[:n]); t_rf = time.perf_counter() - t0
    print(f"n = {n:6d} | fit k-NN: {t_fit * 1000:6.1f} ms | تنبؤ 2000 استعلام: {t_pred * 1000:7.1f} ms | fit غابة 50 شجرة: {t_rf * 1000:8.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

d = np.array([0.0, 2.5, 7.1])          # جار ملاصق تماماً (d = 0)
with np.errstate(divide="ignore"):
    print(f"أوزان 1/d الساذجة: {1.0 / d}")

Xs = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [5.1, 5.2]])
ys = np.array([0, 0, 1, 1])
knn = KNeighborsClassifier(n_neighbors=4, weights="distance").fit(Xs, ys)
print(f"استعلام مطابق لعينة تدريب: {knn.predict_proba(np.array([[0.0, 0.0]]))}")
print(f"استعلام بعيد عن الجميع:     {knn.predict_proba(np.array([[2.4, 2.4]]))}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(n_samples=800, n_features=14, n_informative=5, random_state=3)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"{'k':>4} | {'دقة التدريب':>10} | {'دقة الاختبار':>11}")
for k in [1, 3, 5, 9, 15, 25, 40]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_tr, y_tr)
    print(f"{k:4d} | {pipe.score(X_tr, y_tr):10.4f} | {pipe.score(X_te, y_te):11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)
stds = X.std(axis=0)
share = (stds ** 2) / (stds ** 2).sum()
print(f"عمود proline وحده يحتكر {share[-1]:.1%} من تباين المسافة الإقليدية قبل التطبيع")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
print(f"دقة بدون تطبيع:  {knn_raw.score(X_te, y_te):.4f}")

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
pipe.fit(X_tr, y_tr)
print(f"دقة مع StandardScaler: {pipe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"{'k':>4} | {'uniform':>8} | {'distance':>8}")
for k in range(1, 22, 2):
    accs = []
    for w in ["uniform", "distance"]:
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w))
        pipe.fit(X_tr, y_tr)
        accs.append(pipe.score(X_te, y_te))
    print(f"{k:4d} | {accs[0]:8.4f} | {accs[1]:8.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-54-ch54.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

X, y = make_regression(n_samples=600, n_features=4, noise=25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

cv_scores = {}
for k in range(1, 31, 2):
    p = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    cv_scores[k] = cross_val_score(p, X_tr, y_tr, cv=5,
                                   scoring="neg_root_mean_squared_error").mean()

best_k = max(cv_scores, key=cv_scores.get)
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k))
pipe.fit(X_tr, y_tr)
rmse = root_mean_squared_error(y_te, pipe.predict(X_te))
print(f"أفضل k بالتحقق المتقاطع: {best_k} | RMSE الاختبار: {rmse:.4f}")
