#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنوية SVM ومعاملاتها C و gamma
Mastering scikit-learn 1.9 — SVM

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-52-ch52 (lesson 52)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-52-ch52.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنوية SVM ومعاملاتها C و gamma
Mastering scikit-learn 1.9 — SVM

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-52-ch52 (lesson 52)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import math
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

x = np.array([[2.0, 3.0]])
z = np.array([[1.0, 0.5]])

def phi_quadratic(v):
    v = v.ravel()
    return np.array([v[0] ** 2, v[1] ** 2, np.sqrt(2.0) * v[0] * v[1],
                     np.sqrt(2.0) * v[0], np.sqrt(2.0) * v[1], 1.0])

explicit = float(phi_quadratic(x) @ phi_quadratic(z).T)      # عبر الفضاء المرتفع
kernel = float(polynomial_kernel(x, z, degree=2, gamma=1.0, coef0=1.0)[0, 0])

print(f"φ(x)·φ(z) ببناء الفضاء الستة أبعاد: {explicit:.12f}")
print(f"K(x,z) بالنواة مباشرة:               {kernel:.12f}")
print(f"الفرق: {abs(explicit - kernel):.1e}")
print(f"أبعاد الخريطة التربيعية لبيانات من 100 ميزة: {math.comb(100 + 2, 2)}")
print(f"أبعاد الخريطة الخماسية لبيانات من 100 ميزة: {math.comb(100 + 5, 5)}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

configs = [
    ("linear", dict(kernel="linear", C=10.0)),
    ("poly d=3", dict(kernel="poly", C=10.0, degree=3)),
    ("rbf", dict(kernel="rbf", C=10.0)),
]
for name, params in configs:
    clf = make_pipeline(StandardScaler(), SVC(**params)).fit(X_tr, y_tr)
    print(f"{name:>9} | دقة التدريب: {clf.score(X_tr, y_tr):.4f} | "
          f"دقة الاختبار: {clf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"{'gamma':>8} | {'دقة التدريب':>11} | {'دقة الاختبار':>12} | {'متجهات داعمة':>12}")
print("-" * 54)
for g in [0.01, 0.05, 0.2, 1.0, 5.0, 50.0]:
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma=g)).fit(X_tr, y_tr)
    sv = clf[-1]
    print(f"{g:>8.2f} | {clf.score(X_tr, y_tr):>11.4f} | "
          f"{clf.score(X_te, y_te):>12.4f} | {len(sv.support_vectors_):>12}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=900, n_features=8, n_informative=5,
                           random_state=7)
X[:, 2] *= 500.0         # عمود «بالآلاف»
X[:, 5] *= 0.001         # عمود «بأجزاء الألف»

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(kernel="rbf").fit(X_tr, y_tr)                      # gamma='scale'
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(X_tr, y_tr)

print(f"بلا تطبيع          | دقة: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع StandardScaler  | دقة: {scaled.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(scaled[-1].support_vectors_)}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Cs = [0.1, 1.0, 10.0, 100.0, 1000.0]
Gs = [0.05, 0.2, 1.0, 5.0, 20.0]
scores = np.zeros((len(Gs), len(Cs)))
for i, g in enumerate(Gs):
    for j, c in enumerate(Cs):
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=g))
        scores[i, j] = clf.fit(X_tr, y_tr).score(X_te, y_te)

print("الصفوف: gamma | الأعمدة: C")
print("        " + "".join(f"{c:>8.1f}" for c in Cs))
for i, g in enumerate(Gs):
    print(f"g={g:<6.2f}" + "".join(f"{s:>8.4f}" for s in scores[i, :]))
best = np.unravel_index(scores.argmax(), scores.shape)
print(f"أفضل زوج: C={Cs[best[1]]:g} مع gamma={Gs[best[0]]:g} بدقة اختبار {scores.max():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

X, y = make_classification(n_samples=5000, n_features=20, n_informative=12,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = [
    ("LinearSVC(dual=False)", LinearSVC(C=1.0, dual=False, max_iter=10000)),
    ("SVC(kernel='linear')", SVC(kernel="linear", C=1.0)),
]
for name, model in models:
    pipe = make_pipeline(StandardScaler(), model)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    est = pipe[-1]
    print(f"{name:>22} | زمن: {dt:6.3f}s | دقة: {pipe.score(X_te, y_te):.4f} | "
          f"support_vectors_? {hasattr(est, 'support_vectors_')}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_circles(n_samples=600, noise=0.2, factor=0.4, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

best_g, best_s = None, -1.0
print(f"{'gamma':>8} | {'دقة التدريب':>11} | {'دقة الاختبار':>12}")
for g in [0.01, 0.05, 0.1, 0.5, 2.0, 10.0, 100.0]:
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma=g)).fit(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    if te > best_s:
        best_g, best_s = g, te
    print(f"{g:>8.2f} | {clf.score(X_tr, y_tr):>11.4f} | {te:>12.4f}")
print(f"أفضل gamma = {best_g:g} بدقة اختبار {best_s:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch52.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

for n in [1000, 2000, 4000]:
    X, y = make_classification(n_samples=n, n_features=15, n_informative=8,
                               random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    times = []
    for model in [LinearSVC(C=1.0, dual=False, max_iter=10000),
                  SVC(kernel="linear", C=1.0)]:
        pipe = make_pipeline(StandardScaler(), model)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        times.append(time.perf_counter() - t0)
    print(f"n={n:>5} | LinearSVC: {times[0]:.3f}s | "
          f"SVC(linear): {times[1]:.3f}s | النسبة: {times[1] / times[0]:.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import math
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

x = np.array([[2.0, 3.0]])
z = np.array([[1.0, 0.5]])

def phi_quadratic(v):
    v = v.ravel()
    return np.array([v[0] ** 2, v[1] ** 2, np.sqrt(2.0) * v[0] * v[1],
                     np.sqrt(2.0) * v[0], np.sqrt(2.0) * v[1], 1.0])

explicit = float(phi_quadratic(x) @ phi_quadratic(z).T)      # عبر الفضاء المرتفع
kernel = float(polynomial_kernel(x, z, degree=2, gamma=1.0, coef0=1.0)[0, 0])

print(f"φ(x)·φ(z) ببناء الفضاء الستة أبعاد: {explicit:.12f}")
print(f"K(x,z) بالنواة مباشرة:               {kernel:.12f}")
print(f"الفرق: {abs(explicit - kernel):.1e}")
print(f"أبعاد الخريطة التربيعية لبيانات من 100 ميزة: {math.comb(100 + 2, 2)}")
print(f"أبعاد الخريطة الخماسية لبيانات من 100 ميزة: {math.comb(100 + 5, 5)}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

configs = [
    ("linear", dict(kernel="linear", C=10.0)),
    ("poly d=3", dict(kernel="poly", C=10.0, degree=3)),
    ("rbf", dict(kernel="rbf", C=10.0)),
]
for name, params in configs:
    clf = make_pipeline(StandardScaler(), SVC(**params)).fit(X_tr, y_tr)
    print(f"{name:>9} | دقة التدريب: {clf.score(X_tr, y_tr):.4f} | "
          f"دقة الاختبار: {clf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"{'gamma':>8} | {'دقة التدريب':>11} | {'دقة الاختبار':>12} | {'متجهات داعمة':>12}")
print("-" * 54)
for g in [0.01, 0.05, 0.2, 1.0, 5.0, 50.0]:
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma=g)).fit(X_tr, y_tr)
    sv = clf[-1]
    print(f"{g:>8.2f} | {clf.score(X_tr, y_tr):>11.4f} | "
          f"{clf.score(X_te, y_te):>12.4f} | {len(sv.support_vectors_):>12}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=900, n_features=8, n_informative=5,
                           random_state=7)
X[:, 2] *= 500.0         # عمود «بالآلاف»
X[:, 5] *= 0.001         # عمود «بأجزاء الألف»

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(kernel="rbf").fit(X_tr, y_tr)                      # gamma='scale'
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(X_tr, y_tr)

print(f"بلا تطبيع          | دقة: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع StandardScaler  | دقة: {scaled.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(scaled[-1].support_vectors_)}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Cs = [0.1, 1.0, 10.0, 100.0, 1000.0]
Gs = [0.05, 0.2, 1.0, 5.0, 20.0]
scores = np.zeros((len(Gs), len(Cs)))
for i, g in enumerate(Gs):
    for j, c in enumerate(Cs):
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=g))
        scores[i, j] = clf.fit(X_tr, y_tr).score(X_te, y_te)

print("الصفوف: gamma | الأعمدة: C")
print("        " + "".join(f"{c:>8.1f}" for c in Cs))
for i, g in enumerate(Gs):
    print(f"g={g:<6.2f}" + "".join(f"{s:>8.4f}" for s in scores[i, :]))
best = np.unravel_index(scores.argmax(), scores.shape)
print(f"أفضل زوج: C={Cs[best[1]]:g} مع gamma={Gs[best[0]]:g} بدقة اختبار {scores.max():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

X, y = make_classification(n_samples=5000, n_features=20, n_informative=12,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = [
    ("LinearSVC(dual=False)", LinearSVC(C=1.0, dual=False, max_iter=10000)),
    ("SVC(kernel='linear')", SVC(kernel="linear", C=1.0)),
]
for name, model in models:
    pipe = make_pipeline(StandardScaler(), model)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    est = pipe[-1]
    print(f"{name:>22} | زمن: {dt:6.3f}s | دقة: {pipe.score(X_te, y_te):.4f} | "
          f"support_vectors_? {hasattr(est, 'support_vectors_')}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_circles(n_samples=600, noise=0.2, factor=0.4, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

best_g, best_s = None, -1.0
print(f"{'gamma':>8} | {'دقة التدريب':>11} | {'دقة الاختبار':>12}")
for g in [0.01, 0.05, 0.1, 0.5, 2.0, 10.0, 100.0]:
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma=g)).fit(X_tr, y_tr)
    te = clf.score(X_te, y_te)
    if te > best_s:
        best_g, best_s = g, te
    print(f"{g:>8.2f} | {clf.score(X_tr, y_tr):>11.4f} | {te:>12.4f}")
print(f"أفضل gamma = {best_g:g} بدقة اختبار {best_s:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-52-ch52.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

for n in [1000, 2000, 4000]:
    X, y = make_classification(n_samples=n, n_features=15, n_informative=8,
                               random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    times = []
    for model in [LinearSVC(C=1.0, dual=False, max_iter=10000),
                  SVC(kernel="linear", C=1.0)]:
        pipe = make_pipeline(StandardScaler(), model)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        times.append(time.perf_counter() - t0)
    print(f"n={n:>5} | LinearSVC: {times[0]:.3f}s | "
          f"SVC(linear): {times[1]:.3f}s | النسبة: {times[1] / times[0]:.1f}x")
