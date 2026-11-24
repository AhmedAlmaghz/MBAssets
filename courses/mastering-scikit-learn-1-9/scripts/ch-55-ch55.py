#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — بُنى البحث في الجيران
Mastering scikit-learn 1.9 — Chapter 55 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-55-ch55 (lesson 55)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-55-ch55.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — بُنى البحث في الجيران
Mastering scikit-learn 1.9 — Chapter 55 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-55-ch55 (lesson 55)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(42)
X_warm = rng.uniform(size=(1000, 3)); Q_warm = rng.uniform(size=(20, 3))
for alg in ["brute", "kd_tree", "ball_tree"]:
    NearestNeighbors(algorithm=alg).fit(X_warm).kneighbors(Q_warm)  # تسخين

print(f"{'d':>4} | {'brute':>8} | {'kd_tree':>8} | {'ball_tree':>9}")
for d in [2, 10, 20, 50]:
    X = rng.uniform(0, 1, (20000, d)); Q = rng.uniform(0, 1, (200, d))
    row = []
    for alg in ["brute", "kd_tree", "ball_tree"]:
        best = float("inf")
        for _ in range(3):
            nn = NearestNeighbors(n_neighbors=5, algorithm=alg).fit(X)
            t0 = time.perf_counter(); nn.kneighbors(Q); t = time.perf_counter() - t0
            best = min(best, t)
        row.append(best * 1000)
    print(f"{d:4d} | {row[0]:8.1f} | {row[1]:8.1f} | {row[2]:9.1f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(42)
cases = [
    ("كثيف 100 عينة x 2 خصيصة، k=5",   rng.uniform(size=(100, 2)), None, 5),
    ("كثيف 100 عينة x 20 خصيصة، k=5",  rng.uniform(size=(100, 20)), None, 5),
    ("كثيف 100 عينة x 2، لكن k=50",    rng.uniform(size=(100, 2)), None, 50),
    ("مقياس هافرسين على إحداثيات",     rng.uniform(-np.pi, np.pi, size=(100, 2)), "haversine", 5),
    ("مصفوفة متفرقة 100x20",           sparse.random(100, 20, density=0.1, random_state=42), None, 5),
]
for name, X, metric, k in cases:
    nn = (NearestNeighbors(n_neighbors=k) if metric is None
          else NearestNeighbors(n_neighbors=k, metric=metric))
    print(f"{name:38s} -> {nn.fit(X)._fit_method}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import BallTree

cities = {
    "الرياض": (24.71, 46.68), "جدة": (21.49, 39.19), "دبي": (25.20, 55.27),
    "أبوظبي": (24.45, 54.38), "الدوحة": (25.29, 51.53), "مسقط": (23.59, 58.41),
    "الكويت": (29.38, 47.99), "القاهرة": (30.04, 31.24), "عمان": (31.95, 35.93),
}
names = list(cities)
coords = np.radians(np.array([cities[n] for n in names]))  # العرض ثم الطول
tree = BallTree(coords, metric="haversine")

d, i = tree.query(np.radians([[24.85, 55.10]]), k=3)  # نقطة صحراوية
for dist, idx in zip(d[0], i[0]):
    print(f"{names[idx]:8s} تبعد نحو {dist * 6371.0:6.1f} كم")  # نصف قطر الأرض

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

X, y = make_blobs(n_samples=600, centers=3, cluster_std=[0.6, 0.6, 2.5], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

strict = RadiusNeighborsClassifier(radius=1.5).fit(X_tr, y_tr)
try:
    strict.predict(X_te)
except ValueError as err:
    print(f"بدون outlier_label -> ValueError: {str(err)[:75]}...")

safe = RadiusNeighborsClassifier(radius=1.5, outlier_label="most_frequent").fit(X_tr, y_tr)
print(f"مع outlier_label='most_frequent' -> تنبؤ دون أخطاء على {len(X_te)} عينة")
print(f"دقة النموذج مع النطاق 1.5: {safe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KNeighborsTransformer

X, _ = load_iris(return_X_y=True)
nn = NearestNeighbors(n_neighbors=4).fit(X)
dists, inds = nn.kneighbors(X[:3])
print(f"مؤشرات جيران أول عينتين:\n{inds[:2]}")
print(f"المسافة إلى الجار الأول (النفس!): {dists[:, 0]}")

G = kneighbors_graph(X, n_neighbors=4, mode="distance")
T = (KNeighborsTransformer(n_neighbors=4, mode="distance")
     .set_output(transform="default")  # نريد المصفوفة المتفرقة كما هي
     .fit_transform(X))
print(f"kneighbors_graph: شكل {G.shape}، عناصر غير صفرية: {G.nnz}")
print(f"KNeighborsTransformer: شكل {T.shape}، عناصر غير صفرية: {T.nnz}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.metrics import pairwise_distances_argmin_min

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)  # التطبيع شرط لأي مسافة (الفصل ١٤)
Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
NearestCentroid().fit(Z_tr[:20], y_tr[:20]).predict(Z_te[:5])  # تسخين

for name, mdl in [("NearestCentroid", NearestCentroid()),
                  ("KNeighborsClassifier(k=5)", KNeighborsClassifier(n_neighbors=5))]:
    mdl.fit(Z_tr, y_tr)
    acc = float((mdl.predict(Z_te) == y_te).mean())
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); mdl.predict(Z_te); t = time.perf_counter() - t0
        best = min(best, t)
    print(f"{name:27s} | دقة: {acc:.4f} | تنبؤ {len(Z_te)} عينة: {best * 1000:.2f} ms")

# إعادة بناء التنبؤ يدوياً بالدالة الجديدة: أقرب مركز لكل نقطة
nc = NearestCentroid().fit(Z_tr, y_tr)
idx, dmin = pairwise_distances_argmin_min(Z_te, nc.centroids_)
match = float(np.mean(nc.classes_[idx] == nc.predict(Z_te)))
print(f"تطابق التنبؤ اليدوي (أقرب مركز) مع predict: {match:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for alg in ["brute", "kd_tree"]:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=alg).fit(X_tr, y_tr)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); knn.predict(X_te[:200]); t = time.perf_counter() - t0
        best = min(best, t)
    print(f"algorithm = {alg:8s} | تنبؤ 200 استعلام: {best * 1000:6.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch55.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import RadiusNeighborsClassifier

X, y = make_blobs(n_samples=500, centers=2, cluster_std=1.8, random_state=42)
outliers = np.array([[30.0, 30.0], [-25.0, 28.0], [28.0, -22.0]])
X_all = np.vstack([X, outliers])
y_all = np.concatenate([y, [0, 1, 0]])  # فئات زرعتها للشواذ

strict = RadiusNeighborsClassifier(radius=2.0).fit(X_all, y_all)
try:
    strict.predict(outliers)
except ValueError as err:
    print(f"بدون outlier_label -> ValueError: {str(err)[:70]}...")

safe = RadiusNeighborsClassifier(radius=2.0, outlier_label=-1).fit(X_all, y_all)
print(f"تنبؤ النقاط الشاذة مع outlier_label=-1: {safe.predict(outliers)}")
print(f"عدد العينات التي صنفت شاذة (-1) من كل البيانات: {(safe.predict(X_all) == -1).sum()}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(42)
X_warm = rng.uniform(size=(1000, 3)); Q_warm = rng.uniform(size=(20, 3))
for alg in ["brute", "kd_tree", "ball_tree"]:
    NearestNeighbors(algorithm=alg).fit(X_warm).kneighbors(Q_warm)  # تسخين

print(f"{'d':>4} | {'brute':>8} | {'kd_tree':>8} | {'ball_tree':>9}")
for d in [2, 10, 20, 50]:
    X = rng.uniform(0, 1, (20000, d)); Q = rng.uniform(0, 1, (200, d))
    row = []
    for alg in ["brute", "kd_tree", "ball_tree"]:
        best = float("inf")
        for _ in range(3):
            nn = NearestNeighbors(n_neighbors=5, algorithm=alg).fit(X)
            t0 = time.perf_counter(); nn.kneighbors(Q); t = time.perf_counter() - t0
            best = min(best, t)
        row.append(best * 1000)
    print(f"{d:4d} | {row[0]:8.1f} | {row[1]:8.1f} | {row[2]:9.1f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(42)
cases = [
    ("كثيف 100 عينة x 2 خصيصة، k=5",   rng.uniform(size=(100, 2)), None, 5),
    ("كثيف 100 عينة x 20 خصيصة، k=5",  rng.uniform(size=(100, 20)), None, 5),
    ("كثيف 100 عينة x 2، لكن k=50",    rng.uniform(size=(100, 2)), None, 50),
    ("مقياس هافرسين على إحداثيات",     rng.uniform(-np.pi, np.pi, size=(100, 2)), "haversine", 5),
    ("مصفوفة متفرقة 100x20",           sparse.random(100, 20, density=0.1, random_state=42), None, 5),
]
for name, X, metric, k in cases:
    nn = (NearestNeighbors(n_neighbors=k) if metric is None
          else NearestNeighbors(n_neighbors=k, metric=metric))
    print(f"{name:38s} -> {nn.fit(X)._fit_method}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import BallTree

cities = {
    "الرياض": (24.71, 46.68), "جدة": (21.49, 39.19), "دبي": (25.20, 55.27),
    "أبوظبي": (24.45, 54.38), "الدوحة": (25.29, 51.53), "مسقط": (23.59, 58.41),
    "الكويت": (29.38, 47.99), "القاهرة": (30.04, 31.24), "عمان": (31.95, 35.93),
}
names = list(cities)
coords = np.radians(np.array([cities[n] for n in names]))  # العرض ثم الطول
tree = BallTree(coords, metric="haversine")

d, i = tree.query(np.radians([[24.85, 55.10]]), k=3)  # نقطة صحراوية
for dist, idx in zip(d[0], i[0]):
    print(f"{names[idx]:8s} تبعد نحو {dist * 6371.0:6.1f} كم")  # نصف قطر الأرض

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

X, y = make_blobs(n_samples=600, centers=3, cluster_std=[0.6, 0.6, 2.5], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

strict = RadiusNeighborsClassifier(radius=1.5).fit(X_tr, y_tr)
try:
    strict.predict(X_te)
except ValueError as err:
    print(f"بدون outlier_label -> ValueError: {str(err)[:75]}...")

safe = RadiusNeighborsClassifier(radius=1.5, outlier_label="most_frequent").fit(X_tr, y_tr)
print(f"مع outlier_label='most_frequent' -> تنبؤ دون أخطاء على {len(X_te)} عينة")
print(f"دقة النموذج مع النطاق 1.5: {safe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KNeighborsTransformer

X, _ = load_iris(return_X_y=True)
nn = NearestNeighbors(n_neighbors=4).fit(X)
dists, inds = nn.kneighbors(X[:3])
print(f"مؤشرات جيران أول عينتين:\n{inds[:2]}")
print(f"المسافة إلى الجار الأول (النفس!): {dists[:, 0]}")

G = kneighbors_graph(X, n_neighbors=4, mode="distance")
T = (KNeighborsTransformer(n_neighbors=4, mode="distance")
     .set_output(transform="default")  # نريد المصفوفة المتفرقة كما هي
     .fit_transform(X))
print(f"kneighbors_graph: شكل {G.shape}، عناصر غير صفرية: {G.nnz}")
print(f"KNeighborsTransformer: شكل {T.shape}، عناصر غير صفرية: {T.nnz}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.metrics import pairwise_distances_argmin_min

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)  # التطبيع شرط لأي مسافة (الفصل ١٤)
Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
NearestCentroid().fit(Z_tr[:20], y_tr[:20]).predict(Z_te[:5])  # تسخين

for name, mdl in [("NearestCentroid", NearestCentroid()),
                  ("KNeighborsClassifier(k=5)", KNeighborsClassifier(n_neighbors=5))]:
    mdl.fit(Z_tr, y_tr)
    acc = float((mdl.predict(Z_te) == y_te).mean())
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); mdl.predict(Z_te); t = time.perf_counter() - t0
        best = min(best, t)
    print(f"{name:27s} | دقة: {acc:.4f} | تنبؤ {len(Z_te)} عينة: {best * 1000:.2f} ms")

# إعادة بناء التنبؤ يدوياً بالدالة الجديدة: أقرب مركز لكل نقطة
nc = NearestCentroid().fit(Z_tr, y_tr)
idx, dmin = pairwise_distances_argmin_min(Z_te, nc.centroids_)
match = float(np.mean(nc.classes_[idx] == nc.predict(Z_te)))
print(f"تطابق التنبؤ اليدوي (أقرب مركز) مع predict: {match:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for alg in ["brute", "kd_tree"]:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=alg).fit(X_tr, y_tr)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); knn.predict(X_te[:200]); t = time.perf_counter() - t0
        best = min(best, t)
    print(f"algorithm = {alg:8s} | تنبؤ 200 استعلام: {best * 1000:6.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-55-ch55.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import RadiusNeighborsClassifier

X, y = make_blobs(n_samples=500, centers=2, cluster_std=1.8, random_state=42)
outliers = np.array([[30.0, 30.0], [-25.0, 28.0], [28.0, -22.0]])
X_all = np.vstack([X, outliers])
y_all = np.concatenate([y, [0, 1, 0]])  # فئات زرعتها للشواذ

strict = RadiusNeighborsClassifier(radius=2.0).fit(X_all, y_all)
try:
    strict.predict(outliers)
except ValueError as err:
    print(f"بدون outlier_label -> ValueError: {str(err)[:70]}...")

safe = RadiusNeighborsClassifier(radius=2.0, outlier_label=-1).fit(X_all, y_all)
print(f"تنبؤ النقاط الشاذة مع outlier_label=-1: {safe.predict(outliers)}")
print(f"عدد العينات التي صنفت شاذة (-1) من كل البيانات: {(safe.predict(X_all) == -1).sum()}")
