#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نسخ K-Means العملية
Mastering scikit-learn 1.9 — K-Means

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-61-ch61 (lesson 61)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-61-ch61.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نسخ K-Means العملية
Mastering scikit-learn 1.9 — K-Means

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-61-ch61 (lesson 61)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

X, y = make_blobs(n_samples=200_000, centers=16, n_features=16,
                  cluster_std=0.8, center_box=(-40, 40), random_state=42)

t0 = time.perf_counter()
km = KMeans(n_clusters=16, n_init=3, random_state=42).fit(X)
t_km = time.perf_counter() - t0

t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=16, batch_size=1024, n_init=3, random_state=42).fit(X)
t_mb = time.perf_counter() - t0

print(f"KMeans      | {t_km:6.2f}s | inertia={km.inertia_:12.1f} | ARI مع الحقيقة = {adjusted_rand_score(y, km.labels_):.4f}")
print(f"MiniBatch   | {t_mb:6.2f}s | inertia={mb.inertia_:12.1f} | ARI مع الحقيقة = {adjusted_rand_score(y, mb.labels_):.4f}")
print(f"ARI بين تسميتي النسختين: {adjusted_rand_score(km.labels_, mb.labels_):.4f}")
print(f"تسريع تقريبي: {t_km / t_mb:.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

# بيانات متداخلة عمداً: كثافات متلاصقة في فضاء 8 أبعاد
Xs, ys = make_blobs(n_samples=100_000, centers=12, n_features=8,
                    cluster_std=2.0, center_box=(-12, 12), random_state=42)
for bs in [256, 1024, 4096]:
    t0 = time.perf_counter()
    m = MiniBatchKMeans(n_clusters=12, batch_size=bs, n_init=3, random_state=42).fit(Xs)
    t_mb = time.perf_counter() - t0
    print(f"batch_size={bs:5d} | {t_mb:.2f}s | inertia={m.inertia_:.0f} | ARI={adjusted_rand_score(ys, m.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin, adjusted_rand_score
import numpy as np

X_stream, _ = make_blobs(n_samples=50_000, centers=6, n_features=8,
                         cluster_std=0.9, center_box=(-20, 20), random_state=42)

streamer = MiniBatchKMeans(n_clusters=6, batch_size=2048, random_state=42)
for i, chunk in enumerate(np.array_split(X_stream, 25)):
    streamer.partial_fit(chunk)          # الدفعة تدخل وتُنسى

direct = MiniBatchKMeans(n_clusters=6, batch_size=2048, n_init=3, random_state=42).fit(X_stream)
lab_stream = pairwise_distances_argmin(X_stream, streamer.cluster_centers_)
diff = np.abs(np.sort(streamer.cluster_centers_, axis=0) - np.sort(direct.cluster_centers_, axis=0)).max()
print(f"عولجت 25 دفعة متتالية، الوصول النهائي للمراكز:")
print(f"أقصى فرق بين مراكز التدفق ومراكز التدريب الدفعي: {diff:.4f}")
print(f"ARI بين تسميات التدفق وتسميات التدريب الدفعي: {adjusted_rand_score(direct.labels_, lab_stream):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
import numpy as np

rng = np.random.default_rng(42)
A = rng.normal(loc=[-14, 0], scale=4.0, size=(120, 2))    # واسع قليل النقاط
B = rng.normal(loc=[14, 0], scale=0.6, size=(1400, 2))    # ضيق كثير النقاط
X_bs = np.vstack([A, B])

for strat in ['biggest_inertia', 'largest_cluster']:
    bk = BisectingKMeans(n_clusters=3, bisecting_strategy=strat, random_state=42, n_init=5).fit(X_bs)
    print(f"{strat:16s} (k=3): أحجام العناقيد = {np.bincount(bk.labels_, minlength=3)}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X_h, y_sub = make_blobs(n_samples=1600, centers=[[0, 0], [22, 0], [0, 14], [22, 14]],
                        cluster_std=1.2, random_state=42)
y_macro = (X_h[:, 0] > 11).astype(int)   # التسمية الكبرى: أي نصف المستوى

for k in [2, 3, 4]:
    bk = BisectingKMeans(n_clusters=k, random_state=42, n_init=5).fit(X_h)
    print(f"k={k}: أحجام العناقيد = {np.bincount(bk.labels_, minlength=k)}")

ari_macro = adjusted_rand_score(y_macro, BisectingKMeans(n_clusters=2, random_state=42, n_init=5).fit(X_h).labels_)
ari_sub = adjusted_rand_score(y_sub, BisectingKMeans(n_clusters=4, random_state=42, n_init=5).fit(X_h).labels_)
print(f"k=2 يعيد بنية المجموعتين الكبيرتين (ARI = {ari_macro:.4f})")
print(f"k=4 يعيد الكتل الفرعية الأربع (ARI = {ari_sub:.4f})")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

X, y = make_blobs(n_samples=80_000, centers=10, n_features=6,
                  cluster_std=0.9, center_box=(-30, 30), random_state=42)

t0 = time.perf_counter()
km = KMeans(n_clusters=10, n_init=3, random_state=42).fit(X)
t_km = time.perf_counter() - t0
t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=10, batch_size=1024, n_init=3, random_state=42).fit(X)
t_mb = time.perf_counter() - t0

print(f"KMeans    | {t_km:.2f}s | ARI مع الحقيقة = {adjusted_rand_score(y, km.labels_):.4f}")
print(f"MiniBatch | {t_mb:.2f}s | ARI مع الحقيقة = {adjusted_rand_score(y, mb.labels_):.4f}")
print(f"ARI بين تسميتي النسختين: {adjusted_rand_score(km.labels_, mb.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch61.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

centers = [[0, 0], [0, 6], [0, 12], [24, 0], [24, 6], [24, 12]]
X, y_sub = make_blobs(n_samples=2400, centers=centers, cluster_std=0.9, random_state=42)
y_macro = (X[:, 0] > 12).astype(int)

bk2 = BisectingKMeans(n_clusters=2, random_state=42, n_init=5).fit(X)
bk6 = BisectingKMeans(n_clusters=6, random_state=42, n_init=5).fit(X)
print(f"k=2 : ARI مع التسمية الكبرى = {adjusted_rand_score(y_macro, bk2.labels_):.4f}")
print(f"k=6 : ARI مع التسميات الفرعية = {adjusted_rand_score(y_sub, bk6.labels_):.4f}")
print(f"k=2 : أحجام = {np.bincount(bk2.labels_, minlength=2)}")
print(f"k=6 : أحجام = {np.bincount(bk6.labels_, minlength=6)}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

X, y = make_blobs(n_samples=200_000, centers=16, n_features=16,
                  cluster_std=0.8, center_box=(-40, 40), random_state=42)

t0 = time.perf_counter()
km = KMeans(n_clusters=16, n_init=3, random_state=42).fit(X)
t_km = time.perf_counter() - t0

t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=16, batch_size=1024, n_init=3, random_state=42).fit(X)
t_mb = time.perf_counter() - t0

print(f"KMeans      | {t_km:6.2f}s | inertia={km.inertia_:12.1f} | ARI مع الحقيقة = {adjusted_rand_score(y, km.labels_):.4f}")
print(f"MiniBatch   | {t_mb:6.2f}s | inertia={mb.inertia_:12.1f} | ARI مع الحقيقة = {adjusted_rand_score(y, mb.labels_):.4f}")
print(f"ARI بين تسميتي النسختين: {adjusted_rand_score(km.labels_, mb.labels_):.4f}")
print(f"تسريع تقريبي: {t_km / t_mb:.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

# بيانات متداخلة عمداً: كثافات متلاصقة في فضاء 8 أبعاد
Xs, ys = make_blobs(n_samples=100_000, centers=12, n_features=8,
                    cluster_std=2.0, center_box=(-12, 12), random_state=42)
for bs in [256, 1024, 4096]:
    t0 = time.perf_counter()
    m = MiniBatchKMeans(n_clusters=12, batch_size=bs, n_init=3, random_state=42).fit(Xs)
    t_mb = time.perf_counter() - t0
    print(f"batch_size={bs:5d} | {t_mb:.2f}s | inertia={m.inertia_:.0f} | ARI={adjusted_rand_score(ys, m.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin, adjusted_rand_score
import numpy as np

X_stream, _ = make_blobs(n_samples=50_000, centers=6, n_features=8,
                         cluster_std=0.9, center_box=(-20, 20), random_state=42)

streamer = MiniBatchKMeans(n_clusters=6, batch_size=2048, random_state=42)
for i, chunk in enumerate(np.array_split(X_stream, 25)):
    streamer.partial_fit(chunk)          # الدفعة تدخل وتُنسى

direct = MiniBatchKMeans(n_clusters=6, batch_size=2048, n_init=3, random_state=42).fit(X_stream)
lab_stream = pairwise_distances_argmin(X_stream, streamer.cluster_centers_)
diff = np.abs(np.sort(streamer.cluster_centers_, axis=0) - np.sort(direct.cluster_centers_, axis=0)).max()
print(f"عولجت 25 دفعة متتالية، الوصول النهائي للمراكز:")
print(f"أقصى فرق بين مراكز التدفق ومراكز التدريب الدفعي: {diff:.4f}")
print(f"ARI بين تسميات التدفق وتسميات التدريب الدفعي: {adjusted_rand_score(direct.labels_, lab_stream):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
import numpy as np

rng = np.random.default_rng(42)
A = rng.normal(loc=[-14, 0], scale=4.0, size=(120, 2))    # واسع قليل النقاط
B = rng.normal(loc=[14, 0], scale=0.6, size=(1400, 2))    # ضيق كثير النقاط
X_bs = np.vstack([A, B])

for strat in ['biggest_inertia', 'largest_cluster']:
    bk = BisectingKMeans(n_clusters=3, bisecting_strategy=strat, random_state=42, n_init=5).fit(X_bs)
    print(f"{strat:16s} (k=3): أحجام العناقيد = {np.bincount(bk.labels_, minlength=3)}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X_h, y_sub = make_blobs(n_samples=1600, centers=[[0, 0], [22, 0], [0, 14], [22, 14]],
                        cluster_std=1.2, random_state=42)
y_macro = (X_h[:, 0] > 11).astype(int)   # التسمية الكبرى: أي نصف المستوى

for k in [2, 3, 4]:
    bk = BisectingKMeans(n_clusters=k, random_state=42, n_init=5).fit(X_h)
    print(f"k={k}: أحجام العناقيد = {np.bincount(bk.labels_, minlength=k)}")

ari_macro = adjusted_rand_score(y_macro, BisectingKMeans(n_clusters=2, random_state=42, n_init=5).fit(X_h).labels_)
ari_sub = adjusted_rand_score(y_sub, BisectingKMeans(n_clusters=4, random_state=42, n_init=5).fit(X_h).labels_)
print(f"k=2 يعيد بنية المجموعتين الكبيرتين (ARI = {ari_macro:.4f})")
print(f"k=4 يعيد الكتل الفرعية الأربع (ARI = {ari_sub:.4f})")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import numpy as np, time

X, y = make_blobs(n_samples=80_000, centers=10, n_features=6,
                  cluster_std=0.9, center_box=(-30, 30), random_state=42)

t0 = time.perf_counter()
km = KMeans(n_clusters=10, n_init=3, random_state=42).fit(X)
t_km = time.perf_counter() - t0
t0 = time.perf_counter()
mb = MiniBatchKMeans(n_clusters=10, batch_size=1024, n_init=3, random_state=42).fit(X)
t_mb = time.perf_counter() - t0

print(f"KMeans    | {t_km:.2f}s | ARI مع الحقيقة = {adjusted_rand_score(y, km.labels_):.4f}")
print(f"MiniBatch | {t_mb:.2f}s | ARI مع الحقيقة = {adjusted_rand_score(y, mb.labels_):.4f}")
print(f"ARI بين تسميتي النسختين: {adjusted_rand_score(km.labels_, mb.labels_):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-61-ch61.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

centers = [[0, 0], [0, 6], [0, 12], [24, 0], [24, 6], [24, 12]]
X, y_sub = make_blobs(n_samples=2400, centers=centers, cluster_std=0.9, random_state=42)
y_macro = (X[:, 0] > 12).astype(int)

bk2 = BisectingKMeans(n_clusters=2, random_state=42, n_init=5).fit(X)
bk6 = BisectingKMeans(n_clusters=6, random_state=42, n_init=5).fit(X)
print(f"k=2 : ARI مع التسمية الكبرى = {adjusted_rand_score(y_macro, bk2.labels_):.4f}")
print(f"k=6 : ARI مع التسميات الفرعية = {adjusted_rand_score(y_sub, bk6.labels_):.4f}")
print(f"k=2 : أحجام = {np.bincount(bk2.labels_, minlength=2)}")
print(f"k=6 : أحجام = {np.bincount(bk6.labels_, minlength=6)}")
