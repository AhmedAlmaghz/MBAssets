#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — العناقيد الكثيفة: DBSCAN و OPTICS
Mastering scikit-learn 1.9 — DBSCAN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-63-ch63 (lesson 63)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-63-ch63.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — العناقيد الكثيفة: DBSCAN و OPTICS
Mastering scikit-learn 1.9 — DBSCAN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-63-ch63 (lesson 63)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

for name, maker in [
    ("الهلالان", lambda: make_moons(n_samples=600, noise=0.08, random_state=42)),
    ("الحلقتان", lambda: make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=42)),
]:
    X, y = maker()
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    db = DBSCAN(eps=0.15, min_samples=5).fit(X)
    print(f"{name}: ARI(KMeans)={adjusted_rand_score(y, km.labels_):.4f}"
          f" | ARI(DBSCAN)={adjusted_rand_score(y, db.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np

X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
db = DBSCAN(eps=0.9, min_samples=8).fit(X)
labels = db.labels_

n_core = len(db.core_sample_indices_)
n_noise = int((labels == -1).sum())
n_border = int((labels != -1).sum()) - n_core
print(f"عدد العناقيد: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"نواة: {n_core} | حدّية: {n_border} | ضجيج: {n_noise}")
print(f"شكل components_: {db.components_.shape}")
print(f"أول 10 تسميات: {labels[:10]}")
print(f"أول 6 فهارس نواة: {db.core_sample_indices_[:6]}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.9, random_state=42)
Xs = StandardScaler().fit_transform(X)
for eps in np.arange(0.1, 1.05, 0.1):
    lab = DBSCAN(eps=float(eps), min_samples=5).fit(Xs).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps:.1f} | عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y, lab):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42)
X_units = X.copy()
X_units[:, 0] *= 1000.0   # نفس البيانات لكن ميزة واحدة بوحدة أضخم 1000 مرة

db_raw = DBSCAN(eps=0.5, min_samples=5).fit(X_units)
db_scl = DBSCAN(eps=0.5, min_samples=5).fit(StandardScaler().fit_transform(X_units))
print(f"بلا تطبيع: عناقيد={len(set(db_raw.labels_)) - (1 if -1 in db_raw.labels_ else 0)}"
      f" | ضجيج={(db_raw.labels_ == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y, db_raw.labels_):.3f}")
print(f"مع تطبيع:  عناقيد={len(set(db_scl.labels_)) - (1 if -1 in db_scl.labels_ else 0)}"
      f" | ضجيج={(db_scl.labels_ == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y, db_scl.labels_):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=[500, 500, 90, 90],
                  centers=[[0, 0], [6.5, 0], [0, 9], [6.5, 9]],
                  cluster_std=[0.25, 0.25, 3.0, 3.0], random_state=42)
rng = np.random.default_rng(42)
X_dual = np.vstack([X, rng.uniform(-4, 11, size=(80, 2))])
y_dual = np.concatenate([y, np.full(80, -1)])
mask = y_dual != -1

for eps in [0.5, 1.0, 1.5, 2.0]:
    lab = DBSCAN(eps=eps, min_samples=8).fit(X_dual).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"DBSCAN eps={eps:.1f}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

opt = OPTICS(min_samples=8, cluster_method="dbscan", eps=1.5).fit(X_dual)
lab = opt.labels_
k = len(set(lab)) - (1 if -1 in lab else 0)
print(f"OPTICS dbscan eps=1.5: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

reach = opt.reachability_[opt.ordering_]
finite = reach[np.isfinite(reach)]
print(f"طول متجه اللاح: {len(reach)} | أول قيمة: {reach[0]}")
print(f"متوسط اللاح بعد نقطة البداية: {finite.mean():.3f}")
spikes = np.where(finite > 3.0)[0]
print(f"مواضع القفزات الكبرى (> 3.0): {spikes[:12]}")
print(f"أقصى لاح محدود: {finite.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import time

Xw, _ = make_blobs(n_samples=2000, centers=6, random_state=42)
DBSCAN(eps=0.3, algorithm="brute", n_jobs=1).fit(Xw)   # تسخين
for n in [5000, 10000, 20000]:
    Xb, _ = make_blobs(n_samples=n, centers=6, random_state=42)
    t0 = time.perf_counter()
    DBSCAN(eps=0.3, algorithm="brute", n_jobs=1).fit(Xb)
    t1 = time.perf_counter()
    DBSCAN(eps=0.3, algorithm="kd_tree", n_jobs=1).fit(Xb)
    t2 = time.perf_counter()
    print(f"n={n:6d} | brute={t1 - t0:5.2f}s | kd_tree={t2 - t1:5.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np

X, y = make_moons(n_samples=600, noise=0.15, random_state=42)
for ms in [3, 5, 10, 20]:
    lab = DBSCAN(eps=0.2, min_samples=ms).fit(X).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"min_samples={ms:2d} | عناقيد={k} | ضجيج={(lab == -1).mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part08/ch63.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=42)
for eps in np.arange(0.05, 0.31, 0.05):
    lab = DBSCAN(eps=float(eps), min_samples=5).fit(X).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps:.2f} | عناقيد={k} | ARI={adjusted_rand_score(y, lab):.3f}"
          f" | ضجيج={(lab == -1).mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

for name, maker in [
    ("الهلالان", lambda: make_moons(n_samples=600, noise=0.08, random_state=42)),
    ("الحلقتان", lambda: make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=42)),
]:
    X, y = maker()
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    db = DBSCAN(eps=0.15, min_samples=5).fit(X)
    print(f"{name}: ARI(KMeans)={adjusted_rand_score(y, km.labels_):.4f}"
          f" | ARI(DBSCAN)={adjusted_rand_score(y, db.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np

X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
db = DBSCAN(eps=0.9, min_samples=8).fit(X)
labels = db.labels_

n_core = len(db.core_sample_indices_)
n_noise = int((labels == -1).sum())
n_border = int((labels != -1).sum()) - n_core
print(f"عدد العناقيد: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"نواة: {n_core} | حدّية: {n_border} | ضجيج: {n_noise}")
print(f"شكل components_: {db.components_.shape}")
print(f"أول 10 تسميات: {labels[:10]}")
print(f"أول 6 فهارس نواة: {db.core_sample_indices_[:6]}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.9, random_state=42)
Xs = StandardScaler().fit_transform(X)
for eps in np.arange(0.1, 1.05, 0.1):
    lab = DBSCAN(eps=float(eps), min_samples=5).fit(Xs).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps:.1f} | عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y, lab):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42)
X_units = X.copy()
X_units[:, 0] *= 1000.0   # نفس البيانات لكن ميزة واحدة بوحدة أضخم 1000 مرة

db_raw = DBSCAN(eps=0.5, min_samples=5).fit(X_units)
db_scl = DBSCAN(eps=0.5, min_samples=5).fit(StandardScaler().fit_transform(X_units))
print(f"بلا تطبيع: عناقيد={len(set(db_raw.labels_)) - (1 if -1 in db_raw.labels_ else 0)}"
      f" | ضجيج={(db_raw.labels_ == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y, db_raw.labels_):.3f}")
print(f"مع تطبيع:  عناقيد={len(set(db_scl.labels_)) - (1 if -1 in db_scl.labels_ else 0)}"
      f" | ضجيج={(db_scl.labels_ == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y, db_scl.labels_):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=[500, 500, 90, 90],
                  centers=[[0, 0], [6.5, 0], [0, 9], [6.5, 9]],
                  cluster_std=[0.25, 0.25, 3.0, 3.0], random_state=42)
rng = np.random.default_rng(42)
X_dual = np.vstack([X, rng.uniform(-4, 11, size=(80, 2))])
y_dual = np.concatenate([y, np.full(80, -1)])
mask = y_dual != -1

for eps in [0.5, 1.0, 1.5, 2.0]:
    lab = DBSCAN(eps=eps, min_samples=8).fit(X_dual).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"DBSCAN eps={eps:.1f}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

opt = OPTICS(min_samples=8, cluster_method="dbscan", eps=1.5).fit(X_dual)
lab = opt.labels_
k = len(set(lab)) - (1 if -1 in lab else 0)
print(f"OPTICS dbscan eps=1.5: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

reach = opt.reachability_[opt.ordering_]
finite = reach[np.isfinite(reach)]
print(f"طول متجه اللاح: {len(reach)} | أول قيمة: {reach[0]}")
print(f"متوسط اللاح بعد نقطة البداية: {finite.mean():.3f}")
spikes = np.where(finite > 3.0)[0]
print(f"مواضع القفزات الكبرى (> 3.0): {spikes[:12]}")
print(f"أقصى لاح محدود: {finite.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import time

Xw, _ = make_blobs(n_samples=2000, centers=6, random_state=42)
DBSCAN(eps=0.3, algorithm="brute", n_jobs=1).fit(Xw)   # تسخين
for n in [5000, 10000, 20000]:
    Xb, _ = make_blobs(n_samples=n, centers=6, random_state=42)
    t0 = time.perf_counter()
    DBSCAN(eps=0.3, algorithm="brute", n_jobs=1).fit(Xb)
    t1 = time.perf_counter()
    DBSCAN(eps=0.3, algorithm="kd_tree", n_jobs=1).fit(Xb)
    t2 = time.perf_counter()
    print(f"n={n:6d} | brute={t1 - t0:5.2f}s | kd_tree={t2 - t1:5.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np

X, y = make_moons(n_samples=600, noise=0.15, random_state=42)
for ms in [3, 5, 10, 20]:
    lab = DBSCAN(eps=0.2, min_samples=ms).fit(X).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"min_samples={ms:2d} | عناقيد={k} | ضجيج={(lab == -1).mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-63-ch63.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=42)
for eps in np.arange(0.05, 0.31, 0.05):
    lab = DBSCAN(eps=float(eps), min_samples=5).fit(X).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps:.2f} | عناقيد={k} | ARI={adjusted_rand_score(y, lab):.3f}"
          f" | ضجيج={(lab == -1).mean():.1%}")
