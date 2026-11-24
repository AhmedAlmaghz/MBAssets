#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — K-Means وخوارزمية Lloyd
Mastering scikit-learn 1.9 — K-Means

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-60-ch60 (lesson 60)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-60-ch60.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — K-Means وخوارزمية Lloyd
Mastering scikit-learn 1.9 — K-Means

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-60-ch60 (lesson 60)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, adjusted_rand_score
import numpy as np

X, _ = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=42)
k = 3
rng = np.random.default_rng(42)
centers = X[rng.choice(len(X), size=k, replace=False)]   # بذور عشوائية

for it in range(30):
    assign = pairwise_distances_argmin(X, centers)       # 1) إسناد
    new_centers = np.vstack([X[assign == j].mean(axis=0) for j in range(k)])  # 2) تحديث
    shift = np.abs(new_centers - centers).max()
    centers = new_centers
    if shift < 1e-8:
        break

km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
manual_labels = pairwise_distances_argmin(X, centers)
order = pairwise_distances_argmin(centers, km.cluster_centers_)
print(f"تقاربت النسخة اليدوية بعد {it+1} تكراراً (أكبر إزاحة أخيرة {shift:.1e})")
print(f"أقصى فرق بين مراكزنا اليدوية ومراكز sklearn: {np.abs(centers - km.cluster_centers_[order]).max():.2e}")
print(f"ARI بين تسميات النسخة اليدوية وتسميات sklearn: {adjusted_rand_score(manual_labels, km.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.1, random_state=7)
results = [KMeans(n_clusters=5, init='random', n_init=1, random_state=s).fit(X).inertia_ for s in range(10)]
arr = np.array(results)
print(f"أدنى inertia عبر 10 بذور: {arr.min():.1f}")
print(f"أعلى inertia عبر 10 بذور: {arr.max():.1f}")
print(f"الانحراف المعياري: {arr.std():.1f}")
print(f"القيم العشر: {np.round(arr, 1)}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.1, random_state=7)
for init in ['random', 'k-means++']:
    inertias = np.array([KMeans(n_clusters=5, init=init, n_init=1, random_state=s).fit(X).inertia_ for s in range(20)])
    print(f"init='{init}': أدنى {inertias.min():.1f} | أعلى {inertias.max():.1f} | انحراف {inertias.std():.1f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.9, random_state=42)
print(" k |   inertia   | silhouette")
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"{k:2d} | {km.inertia_:10.1f} | {sil:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import numpy as np

X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print(f"cluster_centers_: {km.cluster_centers_.shape}")
print(f"labels_ أول 10: {km.labels_[:10]}")
print(f"inertia_: {km.inertia_:.2f}")
print(f"n_iter_: {km.n_iter_}")
X_new = np.array([[0.0, 4.0], [-6.0, -2.0], [7.0, 0.0]])
pred = km.predict(X_new)
nearest = euclidean_distances(X_new, km.cluster_centers_).argmin(axis=1)
print(f"predict للنقاط الجديدة: {pred}")
print(f"أقرب مركز حسابياً:      {nearest}")
print(f"predict يطابق أقرب مركز: {(pred == nearest).all()}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np

# (أ) عنقود ضخم بجوار عنقودين صغيرين متقاربين
parts = []
for i, (loc, n_i) in enumerate([([0, 0], 600), ([1.8, 1.8], 60), ([2.6, -0.6], 40)]):
    p = np.random.default_rng(42 + i).normal(loc=loc, scale=0.7, size=(n_i, 2))
    parts.append(p)
X_big = np.vstack(parts)
y_big = np.concatenate([np.zeros(600, dtype=int), np.ones(60, dtype=int), np.full(40, 2)])
lab = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_big)
print(f"أحجام حقيقية 600/60/40 — ARI: {adjusted_rand_score(y_big, lab):.4f}")
print(f"أحجام العناقيد التي وجدها KMeans: {np.bincount(lab, minlength=3)}")

# (ب) ميزة بمقياس ألف ضعف
X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.9, random_state=4)
X_scaled = X.copy()
X_scaled[:, 0] *= 1000.0
ari_raw = adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled))
X_fixed = StandardScaler().fit_transform(X_scaled)
ari_fixed = adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_fixed))
print(f"ARI على البيانات الأصلية:   {adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)):.4f}")
print(f"ARI بعد تضخيم ميزة ×1000:  {ari_raw:.4f}")
print(f"ARI بعد StandardScaler:    {ari_fixed:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print(f"cluster_centers_: {km.cluster_centers_.shape}")
print(f"labels_ أول 8: {km.labels_[:8]}")
print(f"inertia_: {km.inertia_:.2f}")
print(f"n_iter_: {km.n_iter_}")
print(f"ARI مع الأنواع الحقيقية: {adjusted_rand_score(y, km.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch60.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

X, y = load_iris(return_X_y=True)
print(" k |   inertia  | silhouette |  ARI")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    ari = adjusted_rand_score(y, km.labels_)
    print(f"{k:2d} | {km.inertia_:10.1f} | {sil:10.4f} | {ari:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, adjusted_rand_score
import numpy as np

X, _ = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=42)
k = 3
rng = np.random.default_rng(42)
centers = X[rng.choice(len(X), size=k, replace=False)]   # بذور عشوائية

for it in range(30):
    assign = pairwise_distances_argmin(X, centers)       # 1) إسناد
    new_centers = np.vstack([X[assign == j].mean(axis=0) for j in range(k)])  # 2) تحديث
    shift = np.abs(new_centers - centers).max()
    centers = new_centers
    if shift < 1e-8:
        break

km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
manual_labels = pairwise_distances_argmin(X, centers)
order = pairwise_distances_argmin(centers, km.cluster_centers_)
print(f"تقاربت النسخة اليدوية بعد {it+1} تكراراً (أكبر إزاحة أخيرة {shift:.1e})")
print(f"أقصى فرق بين مراكزنا اليدوية ومراكز sklearn: {np.abs(centers - km.cluster_centers_[order]).max():.2e}")
print(f"ARI بين تسميات النسخة اليدوية وتسميات sklearn: {adjusted_rand_score(manual_labels, km.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.1, random_state=7)
results = [KMeans(n_clusters=5, init='random', n_init=1, random_state=s).fit(X).inertia_ for s in range(10)]
arr = np.array(results)
print(f"أدنى inertia عبر 10 بذور: {arr.min():.1f}")
print(f"أعلى inertia عبر 10 بذور: {arr.max():.1f}")
print(f"الانحراف المعياري: {arr.std():.1f}")
print(f"القيم العشر: {np.round(arr, 1)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.1, random_state=7)
for init in ['random', 'k-means++']:
    inertias = np.array([KMeans(n_clusters=5, init=init, n_init=1, random_state=s).fit(X).inertia_ for s in range(20)])
    print(f"init='{init}': أدنى {inertias.min():.1f} | أعلى {inertias.max():.1f} | انحراف {inertias.std():.1f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.9, random_state=42)
print(" k |   inertia   | silhouette")
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"{k:2d} | {km.inertia_:10.1f} | {sil:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import numpy as np

X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print(f"cluster_centers_: {km.cluster_centers_.shape}")
print(f"labels_ أول 10: {km.labels_[:10]}")
print(f"inertia_: {km.inertia_:.2f}")
print(f"n_iter_: {km.n_iter_}")
X_new = np.array([[0.0, 4.0], [-6.0, -2.0], [7.0, 0.0]])
pred = km.predict(X_new)
nearest = euclidean_distances(X_new, km.cluster_centers_).argmin(axis=1)
print(f"predict للنقاط الجديدة: {pred}")
print(f"أقرب مركز حسابياً:      {nearest}")
print(f"predict يطابق أقرب مركز: {(pred == nearest).all()}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np

# (أ) عنقود ضخم بجوار عنقودين صغيرين متقاربين
parts = []
for i, (loc, n_i) in enumerate([([0, 0], 600), ([1.8, 1.8], 60), ([2.6, -0.6], 40)]):
    p = np.random.default_rng(42 + i).normal(loc=loc, scale=0.7, size=(n_i, 2))
    parts.append(p)
X_big = np.vstack(parts)
y_big = np.concatenate([np.zeros(600, dtype=int), np.ones(60, dtype=int), np.full(40, 2)])
lab = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_big)
print(f"أحجام حقيقية 600/60/40 — ARI: {adjusted_rand_score(y_big, lab):.4f}")
print(f"أحجام العناقيد التي وجدها KMeans: {np.bincount(lab, minlength=3)}")

# (ب) ميزة بمقياس ألف ضعف
X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.9, random_state=4)
X_scaled = X.copy()
X_scaled[:, 0] *= 1000.0
ari_raw = adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled))
X_fixed = StandardScaler().fit_transform(X_scaled)
ari_fixed = adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_fixed))
print(f"ARI على البيانات الأصلية:   {adjusted_rand_score(y, KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)):.4f}")
print(f"ARI بعد تضخيم ميزة ×1000:  {ari_raw:.4f}")
print(f"ARI بعد StandardScaler:    {ari_fixed:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print(f"cluster_centers_: {km.cluster_centers_.shape}")
print(f"labels_ أول 8: {km.labels_[:8]}")
print(f"inertia_: {km.inertia_:.2f}")
print(f"n_iter_: {km.n_iter_}")
print(f"ARI مع الأنواع الحقيقية: {adjusted_rand_score(y, km.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-60-ch60.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

X, y = load_iris(return_X_y=True)
print(" k |   inertia  | silhouette |  ARI")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    ari = adjusted_rand_score(y, km.labels_)
    print(f"{k:2d} | {km.inertia_:10.1f} | {sil:10.4f} | {ari:.4f}")
