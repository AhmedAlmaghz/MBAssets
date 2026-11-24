#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التقدير التلقائي: MeanShift و AffinityPropagation
Mastering scikit-learn 1.9 — MeanShift

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-62-ch62 (lesson 62)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-62-ch62.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التقدير التلقائي: MeanShift و AffinityPropagation
Mastering scikit-learn 1.9 — MeanShift

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-62-ch62 (lesson 62)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

Xq, yq = make_blobs(n_samples=400, centers=3, cluster_std=0.9, random_state=42)
for q in [0.05, 0.2, 0.6]:
    bw = estimate_bandwidth(Xq, quantile=q, random_state=42)
    ms = MeanShift(bandwidth=bw).fit(Xq)
    print(f"quantile={q:4.2f} -> bandwidth={bw:6.3f} -> عدد العناقيد = {ms.labels_.max() + 1}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import time

Xb, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)
bw = estimate_bandwidth(Xb, quantile=0.2, random_state=42)

t0 = time.perf_counter()
ms_plain = MeanShift(bandwidth=bw, bin_seeding=False).fit(Xb)
t_plain = time.perf_counter() - t0

t0 = time.perf_counter()
ms_bin = MeanShift(bandwidth=bw, bin_seeding=True).fit(Xb)
t_bin = time.perf_counter() - t0

print(f"bin_seeding=False | {t_plain:6.2f}s | عناقيد = {ms_plain.labels_.max() + 1}")
print(f"bin_seeding=True  | {t_bin:6.2f}s | عناقيد = {ms_bin.labels_.max() + 1}")
print(f"تسريع: {t_plain / t_bin:.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import numpy as np

Xc, _ = make_blobs(n_samples=400, centers=2, cluster_std=0.8, random_state=42)
orphans = np.array([[40.0, 40.0], [-42.0, -38.0]])
X_far = np.vstack([Xc, orphans])
bw = estimate_bandwidth(Xc, quantile=0.3, random_state=42)

ms_all = MeanShift(bandwidth=bw, cluster_all=True, seeds=Xc).fit(X_far)
ms_some = MeanShift(bandwidth=bw, cluster_all=False, seeds=Xc).fit(X_far)
print(f"cluster_all=True : تسميات النقطتين البعيدتين = {ms_all.labels_[-2:]}")
print(f"cluster_all=False: تسميات النقطتين البعيدتين = {ms_some.labels_[-2:]}")
print(f"التسميات الموجودة في الوضع الثاني: {np.unique(ms_some.labels_)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import numpy as np

Xp, yp = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
S = -euclidean_distances(Xp) ** 2
pref_med = float(np.median(S))
print(f"التشابه الوسيط (الافتراضي): {pref_med:.1f}")
for pref in [pref_med, pref_med * 0.5, 0.0]:
    ap = AffinityPropagation(preference=pref, random_state=42, max_iter=500).fit(Xp)
    print(f"preference={pref:8.1f} -> عناقيد = {len(ap.cluster_centers_indices_)} | تكرارات = {ap.n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

Xd, yd = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
for d in [0.5, 0.75, 0.9, 0.95]:
    ap = AffinityPropagation(damping=d, random_state=42, max_iter=1000).fit(Xd)
    print(f"damping={d:4.2f} -> عناقيد = {len(ap.cluster_centers_indices_)}"
          f" | تكرارات حتى التقارب = {ap.n_iter_} | ARI = {adjusted_rand_score(yd, ap.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, AffinityPropagation, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import time

X3, y3 = make_blobs(n_samples=[200, 200, 200], centers=[[0, 0], [7, 2], [2, 12]],
                    cluster_std=[0.4, 1.0, 2.0], random_state=42)
bw = estimate_bandwidth(X3, quantile=0.2, random_state=42)
rows = []

t0 = time.perf_counter()
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X3)
rows.append(("KMeans (k=3)", time.perf_counter() - t0, adjusted_rand_score(y3, km.labels_)))

t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=3, batch_size=256, n_init=3, random_state=42).fit(X3)
rows.append(("MiniBatchKMeans", time.perf_counter() - t0, adjusted_rand_score(y3, mb.labels_)))

t0 = time.perf_counter()
ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X3)
rows.append(("MeanShift (تلقائي)", time.perf_counter() - t0, adjusted_rand_score(y3, ms.labels_)))

t0 = time.perf_counter()
ap = AffinityPropagation(random_state=42, max_iter=500).fit(X3)
rows.append(("AffinityPropagation", time.perf_counter() - t0, adjusted_rand_score(y3, ap.labels_)))

print(f"MeanShift وجد {ms.labels_.max() + 1} عناقيد | AP وجد {len(ap.cluster_centers_indices_)}")
print(f"{'الخوارزمية':20s} | الزمن   | ARI")
for name, t, ari in rows:
    print(f"{name:20s} | {t:6.2f}s | {ari:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

Xe, ye = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42)
for q in [0.05, 0.15, 0.25, 0.35, 0.5]:
    bw = estimate_bandwidth(Xe, quantile=q, random_state=42)
    ms = MeanShift(bandwidth=bw).fit(Xe)
    print(f"quantile={q:4.2f} -> bandwidth={bw:6.3f} -> عناقيد = {ms.labels_.max() + 1}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch62.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances, adjusted_rand_score
import numpy as np

Xi, yi = load_iris(return_X_y=True)
S = -euclidean_distances(Xi) ** 2
pref_med = float(np.median(S))
print(f"التفضيل الافتراضي (الوسيط): {pref_med:.2f}")
for factor in [1.0, 1.5, 2.0, 3.0, 6.0]:
    ap = AffinityPropagation(preference=pref_med * factor, random_state=42, max_iter=500).fit(Xi)
    n_ex = len(ap.cluster_centers_indices_)
    print(f"factor={factor:4.1f} -> نماذج ممثِلة = {n_ex:3d}"
          f" | تكرارات = {ap.n_iter_:3d} | ARI = {adjusted_rand_score(yi, ap.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

Xq, yq = make_blobs(n_samples=400, centers=3, cluster_std=0.9, random_state=42)
for q in [0.05, 0.2, 0.6]:
    bw = estimate_bandwidth(Xq, quantile=q, random_state=42)
    ms = MeanShift(bandwidth=bw).fit(Xq)
    print(f"quantile={q:4.2f} -> bandwidth={bw:6.3f} -> عدد العناقيد = {ms.labels_.max() + 1}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import time

Xb, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)
bw = estimate_bandwidth(Xb, quantile=0.2, random_state=42)

t0 = time.perf_counter()
ms_plain = MeanShift(bandwidth=bw, bin_seeding=False).fit(Xb)
t_plain = time.perf_counter() - t0

t0 = time.perf_counter()
ms_bin = MeanShift(bandwidth=bw, bin_seeding=True).fit(Xb)
t_bin = time.perf_counter() - t0

print(f"bin_seeding=False | {t_plain:6.2f}s | عناقيد = {ms_plain.labels_.max() + 1}")
print(f"bin_seeding=True  | {t_bin:6.2f}s | عناقيد = {ms_bin.labels_.max() + 1}")
print(f"تسريع: {t_plain / t_bin:.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import numpy as np

Xc, _ = make_blobs(n_samples=400, centers=2, cluster_std=0.8, random_state=42)
orphans = np.array([[40.0, 40.0], [-42.0, -38.0]])
X_far = np.vstack([Xc, orphans])
bw = estimate_bandwidth(Xc, quantile=0.3, random_state=42)

ms_all = MeanShift(bandwidth=bw, cluster_all=True, seeds=Xc).fit(X_far)
ms_some = MeanShift(bandwidth=bw, cluster_all=False, seeds=Xc).fit(X_far)
print(f"cluster_all=True : تسميات النقطتين البعيدتين = {ms_all.labels_[-2:]}")
print(f"cluster_all=False: تسميات النقطتين البعيدتين = {ms_some.labels_[-2:]}")
print(f"التسميات الموجودة في الوضع الثاني: {np.unique(ms_some.labels_)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import numpy as np

Xp, yp = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
S = -euclidean_distances(Xp) ** 2
pref_med = float(np.median(S))
print(f"التشابه الوسيط (الافتراضي): {pref_med:.1f}")
for pref in [pref_med, pref_med * 0.5, 0.0]:
    ap = AffinityPropagation(preference=pref, random_state=42, max_iter=500).fit(Xp)
    print(f"preference={pref:8.1f} -> عناقيد = {len(ap.cluster_centers_indices_)} | تكرارات = {ap.n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

Xd, yd = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
for d in [0.5, 0.75, 0.9, 0.95]:
    ap = AffinityPropagation(damping=d, random_state=42, max_iter=1000).fit(Xd)
    print(f"damping={d:4.2f} -> عناقيد = {len(ap.cluster_centers_indices_)}"
          f" | تكرارات حتى التقارب = {ap.n_iter_} | ARI = {adjusted_rand_score(yd, ap.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, AffinityPropagation, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import time

X3, y3 = make_blobs(n_samples=[200, 200, 200], centers=[[0, 0], [7, 2], [2, 12]],
                    cluster_std=[0.4, 1.0, 2.0], random_state=42)
bw = estimate_bandwidth(X3, quantile=0.2, random_state=42)
rows = []

t0 = time.perf_counter()
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X3)
rows.append(("KMeans (k=3)", time.perf_counter() - t0, adjusted_rand_score(y3, km.labels_)))

t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=3, batch_size=256, n_init=3, random_state=42).fit(X3)
rows.append(("MiniBatchKMeans", time.perf_counter() - t0, adjusted_rand_score(y3, mb.labels_)))

t0 = time.perf_counter()
ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X3)
rows.append(("MeanShift (تلقائي)", time.perf_counter() - t0, adjusted_rand_score(y3, ms.labels_)))

t0 = time.perf_counter()
ap = AffinityPropagation(random_state=42, max_iter=500).fit(X3)
rows.append(("AffinityPropagation", time.perf_counter() - t0, adjusted_rand_score(y3, ap.labels_)))

print(f"MeanShift وجد {ms.labels_.max() + 1} عناقيد | AP وجد {len(ap.cluster_centers_indices_)}")
print(f"{'الخوارزمية':20s} | الزمن   | ARI")
for name, t, ari in rows:
    print(f"{name:20s} | {t:6.2f}s | {ari:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

Xe, ye = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42)
for q in [0.05, 0.15, 0.25, 0.35, 0.5]:
    bw = estimate_bandwidth(Xe, quantile=q, random_state=42)
    ms = MeanShift(bandwidth=bw).fit(Xe)
    print(f"quantile={q:4.2f} -> bandwidth={bw:6.3f} -> عناقيد = {ms.labels_.max() + 1}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-62-ch62.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances, adjusted_rand_score
import numpy as np

Xi, yi = load_iris(return_X_y=True)
S = -euclidean_distances(Xi) ** 2
pref_med = float(np.median(S))
print(f"التفضيل الافتراضي (الوسيط): {pref_med:.2f}")
for factor in [1.0, 1.5, 2.0, 3.0, 6.0]:
    ap = AffinityPropagation(preference=pref_med * factor, random_state=42, max_iter=500).fit(Xi)
    n_ex = len(ap.cluster_centers_indices_)
    print(f"factor={factor:4.1f} -> نماذج ممثِلة = {n_ex:3d}"
          f" | تكرارات = {ap.n_iter_:3d} | ARI = {adjusted_rand_score(yi, ap.labels_):.4f}")
