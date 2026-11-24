#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التعلّم غير الخاضع للرقابة: الخريطة
Mastering scikit-learn 1.9 — Chapter 59 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-59-ch59 (lesson 59)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-59-ch59.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التعلّم غير الخاضع للرقابة: الخريطة
Mastering scikit-learn 1.9 — Chapter 59 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-59-ch59 (lesson 59)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
print("أول 8 تسميات من fit_predict:", kmeans.fit_predict(X)[:8])
print("شكل المراكز cluster_centers_:", kmeans.cluster_centers_.shape)
print(f"مجموع مربعات المسافات inertia: {kmeans.inertia_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X_iris, y_iris = load_iris(return_X_y=True)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_iris)
print(f"الأبعاد قبل التحويل: {X_iris.shape} — بعد fit_transform: {X_2d.shape}")
print(f"نسبة التباين المفسَّر بالمكوّنين معاً: {pca.explained_variance_ratio_.sum():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest
import numpy as np

rng = np.random.default_rng(42)
X_norm = rng.normal(loc=[0, 0], scale=1.0, size=(200, 2))
X_out = rng.uniform(low=6.0, high=9.0, size=(10, 2))
X_mix = np.vstack([X_norm, X_out])

iso = IsolationForest(n_estimators=100, random_state=42).fit(X_mix)
scores = iso.score_samples(X_mix)
print(f"متوسط درجة الكثافة للعينات العادية:  {scores[:200].mean():.4f}")
print(f"متوسط درجة الكثافة للشواذ المحقونة: {scores[200:].mean():.4f}")
idx_low = scores.argsort()[:10]
print(f"من بين أدنى 10 درجات، كم نقطة شاذة محقونة؟ {int((idx_low >= 200).sum())} من 10")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KernelDensity
import numpy as np

rng = np.random.default_rng(42)
X_dens = rng.normal(loc=0.0, scale=1.0, size=(400, 1))
kd = KernelDensity(bandwidth=0.3).fit(X_dens)
log_d_center = kd.score_samples([[0.0]])   # قلب السحابة
log_d_far = kd.score_samples([[4.0]])      # هامش بعيد
print(f"لوغاريتم الكثافة المقدَّرة عند القلب 0.0: {log_d_center[0]:.4f}")
print(f"لوغاريتم الكثافة المقدَّرة عند الهامش 4.0: {log_d_far[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X_moons, y_true = make_moons(n_samples=400, noise=0.06, random_state=42)
labels_km = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_moons)
ari = adjusted_rand_score(y_true, labels_km)
print(f"طلبنا من KMeans عنقودين، فأعاد 2: {(labels_km.max()+1)==2}")
print(f"ARI بين تسميات KMeans والهلالين الحقيقيين: {ari:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

X5, _ = make_blobs(n_samples=400, centers=5, cluster_std=0.7, random_state=42)
for k in [3, 5, 8]:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X5)
    print(f"KMeans بـ k={k}: inertia = {km_k.inertia_:.1f}")
bw = estimate_bandwidth(X5, quantile=0.15, random_state=42)
ms = MeanShift(bandwidth=bw).fit(X5)
print(f"MeanShift اختار بنفسه {ms.labels_.max() + 1} عناقيد بلا أن نطلب رقماً")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)

labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
iso = IsolationForest(n_estimators=100, random_state=42).fit(X)
strange = iso.score_samples(X)

print(f"(أ) التجميع أجاب: كل نقطة الآن لها تسمية من 0 إلى {labels.max()}")
print(f"(ب) تقليل الأبعاد أجاب: {X.shape[1]} أعمدة صارت {X_2d.shape[1]} إحداثيات")
print(f"(ج) كشف الشذوذ أجاب: أدنى درجة غرابة {strange.min():.4f} عند العينة {int(np.argmin(strange))}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch59.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X3, y3 = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=42)
for k in [2, 4]:
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X3)
    print(f"k={k}: inertia={KMeans(n_clusters=k, n_init=10, random_state=42).fit(X3).inertia_:.1f}"
          f" | ARI مع الحقيقة = {adjusted_rand_score(y3, lab):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
print("أول 8 تسميات من fit_predict:", kmeans.fit_predict(X)[:8])
print("شكل المراكز cluster_centers_:", kmeans.cluster_centers_.shape)
print(f"مجموع مربعات المسافات inertia: {kmeans.inertia_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X_iris, y_iris = load_iris(return_X_y=True)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_iris)
print(f"الأبعاد قبل التحويل: {X_iris.shape} — بعد fit_transform: {X_2d.shape}")
print(f"نسبة التباين المفسَّر بالمكوّنين معاً: {pca.explained_variance_ratio_.sum():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest
import numpy as np

rng = np.random.default_rng(42)
X_norm = rng.normal(loc=[0, 0], scale=1.0, size=(200, 2))
X_out = rng.uniform(low=6.0, high=9.0, size=(10, 2))
X_mix = np.vstack([X_norm, X_out])

iso = IsolationForest(n_estimators=100, random_state=42).fit(X_mix)
scores = iso.score_samples(X_mix)
print(f"متوسط درجة الكثافة للعينات العادية:  {scores[:200].mean():.4f}")
print(f"متوسط درجة الكثافة للشواذ المحقونة: {scores[200:].mean():.4f}")
idx_low = scores.argsort()[:10]
print(f"من بين أدنى 10 درجات، كم نقطة شاذة محقونة؟ {int((idx_low >= 200).sum())} من 10")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KernelDensity
import numpy as np

rng = np.random.default_rng(42)
X_dens = rng.normal(loc=0.0, scale=1.0, size=(400, 1))
kd = KernelDensity(bandwidth=0.3).fit(X_dens)
log_d_center = kd.score_samples([[0.0]])   # قلب السحابة
log_d_far = kd.score_samples([[4.0]])      # هامش بعيد
print(f"لوغاريتم الكثافة المقدَّرة عند القلب 0.0: {log_d_center[0]:.4f}")
print(f"لوغاريتم الكثافة المقدَّرة عند الهامش 4.0: {log_d_far[0]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X_moons, y_true = make_moons(n_samples=400, noise=0.06, random_state=42)
labels_km = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_moons)
ari = adjusted_rand_score(y_true, labels_km)
print(f"طلبنا من KMeans عنقودين، فأعاد 2: {(labels_km.max()+1)==2}")
print(f"ARI بين تسميات KMeans والهلالين الحقيقيين: {ari:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

X5, _ = make_blobs(n_samples=400, centers=5, cluster_std=0.7, random_state=42)
for k in [3, 5, 8]:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X5)
    print(f"KMeans بـ k={k}: inertia = {km_k.inertia_:.1f}")
bw = estimate_bandwidth(X5, quantile=0.15, random_state=42)
ms = MeanShift(bandwidth=bw).fit(X5)
print(f"MeanShift اختار بنفسه {ms.labels_.max() + 1} عناقيد بلا أن نطلب رقماً")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)

labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
iso = IsolationForest(n_estimators=100, random_state=42).fit(X)
strange = iso.score_samples(X)

print(f"(أ) التجميع أجاب: كل نقطة الآن لها تسمية من 0 إلى {labels.max()}")
print(f"(ب) تقليل الأبعاد أجاب: {X.shape[1]} أعمدة صارت {X_2d.shape[1]} إحداثيات")
print(f"(ج) كشف الشذوذ أجاب: أدنى درجة غرابة {strange.min():.4f} عند العينة {int(np.argmin(strange))}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-59-ch59.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X3, y3 = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=42)
for k in [2, 4]:
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X3)
    print(f"k={k}: inertia={KMeans(n_clusters=k, n_init=10, random_state=42).fit(X3).inertia_:.1f}"
          f" | ARI مع الحقيقة = {adjusted_rand_score(y3, lab):.4f}")
