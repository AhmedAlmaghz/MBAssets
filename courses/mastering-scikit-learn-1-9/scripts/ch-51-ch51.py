#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — آلات المتجهات الداعمة: الحيلة الحقيقية
Mastering scikit-learn 1.9 — Chapter 51 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-51-ch51 (lesson 51)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-51-ch51.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — آلات المتجهات الداعمة: الحيلة الحقيقية
Mastering scikit-learn 1.9 — Chapter 51 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-51-ch51 (lesson 51)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(42)
inner = rng.normal([0.0, 0.0], 0.35, (60, 2))         # فئة «نوم جيد»: جزيرة وسط
r = 2.6 + rng.normal(0.0, 0.22, 90)
theta = rng.uniform(0.0, 2.0 * np.pi, 90)
ring = np.c_[r * np.cos(theta), r * np.sin(theta)]    # فئة «منبّه متأخر»: حلقة
far = rng.normal([6.0, 6.0], 0.55, (60, 2))           # فئة «لا نوم»: ركن بعيد
X = np.vstack([inner, ring, far])
y = np.r_[np.zeros(60, int), np.ones(90, int), 2 * np.ones(60, int)]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

linear = make_pipeline(StandardScaler(), SVC(kernel="linear", C=10)).fit(X_tr, y_tr)
rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.3)).fit(X_tr, y_tr)

print(f"دقة الحد المستقيم (kernel=linear): {linear.score(X_te, y_te):.4f}")
print(f"دقة الحد المنحني  (kernel=rbf):    {rbf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.9, random_state=42)
clf = SVC(kernel="linear", C=1e6).fit(X, y)          # C ضخم = هامش صلب عملياً

w = clf.coef_[0]
b = clf.intercept_[0]
norm_w = np.linalg.norm(w)
d_sv = np.abs(clf.support_vectors_ @ w + b) / norm_w  # مسافات الداعمين للمستوى

print(f"متجهات داعمة: {len(clf.support_vectors_)} من أصل {len(X)} عينة")
print(f"||w|| = {norm_w:.6f}")
print(f"الهامش الكامل 2/||w|| = {2.0 / norm_w:.6f}")
print(f"أصغر مسافة داعم إلى المستوى: {d_sv.min():.6f} (المتوقع 1/||w|| = {1.0 / norm_w:.6f})")
print(f"أكبر مسافة داعم إلى المستوى: {d_sv.max():.6f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_blobs(n_samples=600, centers=[[-2.0, 0.0], [2.0, 0.0]],
                  cluster_std=1.4, random_state=42)     # فئتان متداخلتان جزئياً
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"{'C':>8} | {'متجهات داعمة':>12} | {'دقة التدريب':>11} | {'دقة الاختبار':>12}")
print("-" * 56)
for C in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0]:
    clf = SVC(kernel="linear", C=C).fit(X_tr, y_tr)
    print(f"{C:>8.2f} | {len(clf.support_vectors_):>12} | "
          f"{clf.score(X_tr, y_tr):>11.4f} | {clf.score(X_te, y_te):>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_blobs(n_samples=800, centers=[[-2.2, 0.0], [2.2, 0.0]],
                  cluster_std=1.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

full = SVC(kernel="linear", C=1.0).fit(X_tr, y_tr)
sv_idx = full.support_                                # فهارس متجهات الدعم
trimmed = SVC(kernel="linear", C=1.0).fit(X_tr[sv_idx], y_tr[sv_idx])

d_full = full.decision_function(X_te)
d_trim = trimmed.decision_function(X_te)
print(f"عينات تدريب: {len(X_tr)} | متجهات داعمة: {len(sv_idx)} "
      f"({len(sv_idx) / len(X_tr):.1%} فقط)")
print(f"أقصى فرق في قيم القرار بعد حذف غير الداعمين: {np.abs(d_full - d_trim).max():.2e}")
print(f"تطابق تصنيفات الاختبار: {(full.predict(X_te) == trimmed.predict(X_te)).all()}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.5)).fit(X_tr, y_tr)
svc = clf[-1]                                         # مقدر SVC داخل الأنبوب
K = len(svc.classes_)

print(f"الفئات: {svc.classes_}")
print(f"عدد المصنفات الزوجية OvO: {K * (K - 1) // 2}")
print(f"n_support_ لكل فئة: {svc.n_support_} (مجموعها {svc.n_support_.sum()})")
print(f"شكل support_vectors_: {svc.support_vectors_.shape}")
print(f"شكل decision_function للاختبار: {clf.decision_function(X_te).shape}")
print(f"دقة الاختبار: {clf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
                           random_state=42)
X[:, 0] *= 900.0        # عمود بوحدة «ألف»
X[:, 1] *= 0.0001       # عمود بوحدة «جزء من ألف»

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(kernel="rbf", gamma=0.05).fit(X_tr, y_tr)
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=0.5)).fit(X_tr, y_tr)

print(f"بلا تحجيم          | دقة: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع StandardScaler  | دقة: {scaled.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(scaled[-1].support_vectors_)}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(C=1.0).fit(X_tr, y_tr)                     # kernel=rbf الافتراضي
pipe = make_pipeline(StandardScaler(), SVC(C=1.0)).fit(X_tr, y_tr)
sv = pipe[-1]

print(f"بلا تحجيم | دقة الاختبار: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع تحجيم  | دقة الاختبار: {pipe.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(sv.support_vectors_)}")
print(f"n_support_ لكل فئة: {sv.n_support_} (خبيث/حميد)")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch51.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=300, centers=2, cluster_std=0.9, random_state=42)
clf = SVC(kernel="linear", C=1e5).fit(X, y)
w = clf.coef_[0]
norm_w = np.linalg.norm(w)
d_sv = np.abs(clf.support_vectors_ @ w + clf.intercept_[0]) / norm_w

print(f"متجهات داعمة: {len(clf.support_vectors_)} من أصل {len(X)}")
print(f"الهامش اليدوي 2/||w|| = {2.0 / norm_w:.6f}")
print(f"مسافات الداعمين: [{d_sv.min():.6f}، {d_sv.max():.6f}] "
      f"(المتوقع 1/||w|| = {1.0 / norm_w:.6f})")

mask = np.ones(len(X), dtype=bool)
mask[clf.support_] = False                        # احذف غير الداعمين فقط
trimmed = SVC(kernel="linear", C=1e5).fit(X[mask], y[mask])
agree = (clf.predict(X) == trimmed.predict(X)).mean()
print(f"بعد حذف {mask.sum()} صفاً غير داعم وإعادة التدريب: "
      f"تطابق التصنيفات على كل العينات = {agree:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(42)
inner = rng.normal([0.0, 0.0], 0.35, (60, 2))         # فئة «نوم جيد»: جزيرة وسط
r = 2.6 + rng.normal(0.0, 0.22, 90)
theta = rng.uniform(0.0, 2.0 * np.pi, 90)
ring = np.c_[r * np.cos(theta), r * np.sin(theta)]    # فئة «منبّه متأخر»: حلقة
far = rng.normal([6.0, 6.0], 0.55, (60, 2))           # فئة «لا نوم»: ركن بعيد
X = np.vstack([inner, ring, far])
y = np.r_[np.zeros(60, int), np.ones(90, int), 2 * np.ones(60, int)]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

linear = make_pipeline(StandardScaler(), SVC(kernel="linear", C=10)).fit(X_tr, y_tr)
rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.3)).fit(X_tr, y_tr)

print(f"دقة الحد المستقيم (kernel=linear): {linear.score(X_te, y_te):.4f}")
print(f"دقة الحد المنحني  (kernel=rbf):    {rbf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.9, random_state=42)
clf = SVC(kernel="linear", C=1e6).fit(X, y)          # C ضخم = هامش صلب عملياً

w = clf.coef_[0]
b = clf.intercept_[0]
norm_w = np.linalg.norm(w)
d_sv = np.abs(clf.support_vectors_ @ w + b) / norm_w  # مسافات الداعمين للمستوى

print(f"متجهات داعمة: {len(clf.support_vectors_)} من أصل {len(X)} عينة")
print(f"||w|| = {norm_w:.6f}")
print(f"الهامش الكامل 2/||w|| = {2.0 / norm_w:.6f}")
print(f"أصغر مسافة داعم إلى المستوى: {d_sv.min():.6f} (المتوقع 1/||w|| = {1.0 / norm_w:.6f})")
print(f"أكبر مسافة داعم إلى المستوى: {d_sv.max():.6f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_blobs(n_samples=600, centers=[[-2.0, 0.0], [2.0, 0.0]],
                  cluster_std=1.4, random_state=42)     # فئتان متداخلتان جزئياً
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"{'C':>8} | {'متجهات داعمة':>12} | {'دقة التدريب':>11} | {'دقة الاختبار':>12}")
print("-" * 56)
for C in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0]:
    clf = SVC(kernel="linear", C=C).fit(X_tr, y_tr)
    print(f"{C:>8.2f} | {len(clf.support_vectors_):>12} | "
          f"{clf.score(X_tr, y_tr):>11.4f} | {clf.score(X_te, y_te):>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_blobs(n_samples=800, centers=[[-2.2, 0.0], [2.2, 0.0]],
                  cluster_std=1.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

full = SVC(kernel="linear", C=1.0).fit(X_tr, y_tr)
sv_idx = full.support_                                # فهارس متجهات الدعم
trimmed = SVC(kernel="linear", C=1.0).fit(X_tr[sv_idx], y_tr[sv_idx])

d_full = full.decision_function(X_te)
d_trim = trimmed.decision_function(X_te)
print(f"عينات تدريب: {len(X_tr)} | متجهات داعمة: {len(sv_idx)} "
      f"({len(sv_idx) / len(X_tr):.1%} فقط)")
print(f"أقصى فرق في قيم القرار بعد حذف غير الداعمين: {np.abs(d_full - d_trim).max():.2e}")
print(f"تطابق تصنيفات الاختبار: {(full.predict(X_te) == trimmed.predict(X_te)).all()}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma=0.5)).fit(X_tr, y_tr)
svc = clf[-1]                                         # مقدر SVC داخل الأنبوب
K = len(svc.classes_)

print(f"الفئات: {svc.classes_}")
print(f"عدد المصنفات الزوجية OvO: {K * (K - 1) // 2}")
print(f"n_support_ لكل فئة: {svc.n_support_} (مجموعها {svc.n_support_.sum()})")
print(f"شكل support_vectors_: {svc.support_vectors_.shape}")
print(f"شكل decision_function للاختبار: {clf.decision_function(X_te).shape}")
print(f"دقة الاختبار: {clf.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
                           random_state=42)
X[:, 0] *= 900.0        # عمود بوحدة «ألف»
X[:, 1] *= 0.0001       # عمود بوحدة «جزء من ألف»

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(kernel="rbf", gamma=0.05).fit(X_tr, y_tr)
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=0.5)).fit(X_tr, y_tr)

print(f"بلا تحجيم          | دقة: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع StandardScaler  | دقة: {scaled.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(scaled[-1].support_vectors_)}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

raw = SVC(C=1.0).fit(X_tr, y_tr)                     # kernel=rbf الافتراضي
pipe = make_pipeline(StandardScaler(), SVC(C=1.0)).fit(X_tr, y_tr)
sv = pipe[-1]

print(f"بلا تحجيم | دقة الاختبار: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw.support_vectors_)}")
print(f"مع تحجيم  | دقة الاختبار: {pipe.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(sv.support_vectors_)}")
print(f"n_support_ لكل فئة: {sv.n_support_} (خبيث/حميد)")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-51-ch51.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=300, centers=2, cluster_std=0.9, random_state=42)
clf = SVC(kernel="linear", C=1e5).fit(X, y)
w = clf.coef_[0]
norm_w = np.linalg.norm(w)
d_sv = np.abs(clf.support_vectors_ @ w + clf.intercept_[0]) / norm_w

print(f"متجهات داعمة: {len(clf.support_vectors_)} من أصل {len(X)}")
print(f"الهامش اليدوي 2/||w|| = {2.0 / norm_w:.6f}")
print(f"مسافات الداعمين: [{d_sv.min():.6f}، {d_sv.max():.6f}] "
      f"(المتوقع 1/||w|| = {1.0 / norm_w:.6f})")

mask = np.ones(len(X), dtype=bool)
mask[clf.support_] = False                        # احذف غير الداعمين فقط
trimmed = SVC(kernel="linear", C=1e5).fit(X[mask], y[mask])
agree = (clf.predict(X) == trimmed.predict(X)).mean()
print(f"بعد حذف {mask.sum()} صفاً غير داعم وإعادة التدريب: "
      f"تطابق التصنيفات على كل العينات = {agree:.1%}")
