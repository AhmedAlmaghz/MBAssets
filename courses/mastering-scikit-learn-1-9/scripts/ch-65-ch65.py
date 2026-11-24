#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التجميع الهرمي والطوبولوجي
Mastering scikit-learn 1.9 — Chapter 65 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-65-ch65 (lesson 65)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-65-ch65.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التجميع الهرمي والطوبولوجي
Mastering scikit-learn 1.9 — Chapter 65 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-65-ch65 (lesson 65)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

datasets = {
    "الهلالان": make_moons(n_samples=600, noise=0.05, random_state=42),
    "الحلقتان": make_circles(n_samples=600, factor=0.45, noise=0.05, random_state=42),
    "الكرات": make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=42),
}
for name, (X, y) in datasets.items():
    row = [f"{name:8s}"]
    for lk in ["ward", "complete", "average", "single"]:
        lab = AgglomerativeClustering(n_clusters=2 if name != "الكرات" else 3,
                                      linkage=lk).fit_predict(X)
        row.append(f"{lk}={adjusted_rand_score(y, lab):.3f}")
    print(" | ".join(row))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_moons(n_samples=150, noise=0.05, random_state=42)
Z = linkage(X, method="ward")
print(f"شكل Z: {Z.shape} (سطر لكل اندماج من أصل {len(X) - 1})")
print(f"أول اندماج: {np.round(Z[0], 3)}")
print(f"آخر ثلاثة ارتفاعات: {np.round(Z[-3:, 2], 2)}")

R = dendrogram(Z, no_plot=True)
print(f"أوراق الدندروغرام: {len(R['leaves'])}")

for t in [2.0, 3.0, 4.0, 6.0]:
    fc = fcluster(Z, t=t, criterion="distance")
    print(f"قطع عند ارتفاع {t}: {len(np.unique(fc))} عناقيد")

ag = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
fc = fcluster(Z, t=2, criterion="maxclust")
print(f"fcluster(maxclust=2) مقابل sklearn: ARI={adjusted_rand_score(ag, fc):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=42)
for th in [5, 8, 12, 18, 25]:
    ag = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                 linkage="ward", compute_full_tree=True).fit(X)
    print(f"حد={th:2d} | عناقيد={ag.n_clusters_} | اندماجات={len(ag.distances_)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import coo_matrix
import numpy as np

rng = np.random.default_rng(42)
h, w = 40, 40
img = np.zeros((h, w))
img[:20, :20] = rng.normal(2.0, 1.2, size=(20, 20))
img[:20, 20:] = rng.normal(8.0, 1.2, size=(20, 20))
img[20:, :20] = rng.normal(8.0, 1.2, size=(20, 20))
img[20:, 20:] = rng.normal(2.0, 1.2, size=(20, 20))
y_img = np.zeros((h, w), dtype=int)
y_img[:20, 20:] = 1
y_img[20:, :20] = 2
y_img[20:, 20:] = 3
y_img = y_img.ravel()

i = np.arange(h * w)
r_i, c_i = np.divmod(i, w)
src = np.r_[i[c_i < w - 1], i[r_i < h - 1]]
dst = np.r_[i[c_i < w - 1] + 1, i[r_i < h - 1] + w]
conn = coo_matrix((np.ones(len(src)), (src, dst)), shape=(h * w, h * w)).tocsr()
conn = conn + conn.T
print(f"حواف شبكة الجيران: {conn.nnz // 2}")

X_pix = img.ravel().reshape(-1, 1)
lab_c = AgglomerativeClustering(n_clusters=4, linkage="ward",
                                connectivity=conn).fit_predict(X_pix)
lab_f = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X_pix)

def contiguity(lab):
    L = lab.reshape(h, w)
    return ((L[:, :-1] == L[:, 1:]).mean() + (L[:-1, :] == L[1:, :]).mean()) / 2

print(f"بقيد الاتصال: ARI={adjusted_rand_score(y_img, lab_c):.3f} | تماسك مكاني={contiguity(lab_c):.1%}")
print(f"بلا قيد:      ARI={adjusted_rand_score(y_img, lab_f):.3f} | تماسك مكاني={contiguity(lab_f):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3,
                                              random_state=42, stratify=yd)
clf = RidgeClassifier().fit(Xd_tr, yd_tr)
print(f"RidgeClassifier على 64 عموداً: الدقة={accuracy_score(yd_te, clf.predict(Xd_te)):.4f}")

for k in [32, 16]:
    pipe = make_pipeline(FeatureAgglomeration(n_clusters=k), RidgeClassifier())
    pipe.fit(Xd_tr, yd_tr)
    acc = accuracy_score(yd_te, pipe.predict(Xd_te))
    print(f"FeatureAgglomeration({k}) + RidgeClassifier: الدقة={acc:.4f}")

fa = FeatureAgglomeration(n_clusters=4).fit(Xd_tr)
print(f"مجموعات الأعمدة (أول 20 بكسل): {fa.labels_[:20]}")
Xt = np.asarray(fa.transform(Xd_tr))
print(f"شكل التحويل: {Xt.shape} | العودة: {fa.inverse_transform(Xt).shape}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

Xm, ym = make_moons(n_samples=600, noise=0.05, random_state=42)
Xc, yc = make_circles(n_samples=600, factor=0.45, noise=0.05, random_state=42)
lab_m = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                           n_neighbors=20, random_state=42).fit_predict(Xm)
lab_c = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                           n_neighbors=40, random_state=42).fit_predict(Xc)
print(f"الهلالان: ARI={adjusted_rand_score(ym, lab_m):.4f}")
print(f"الحلقتان: ARI={adjusted_rand_score(yc, lab_c):.4f}")

for al in ["kmeans", "discretize", "cluster_qr"]:
    lab = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                             n_neighbors=20, assign_labels=al,
                             random_state=42).fit_predict(Xm)
    print(f"assign_labels={al:10s}: ARI={adjusted_rand_score(ym, lab):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
import time

for n in [1200, 3600]:
    Xn, _ = make_blobs(n_samples=n, centers=3, random_state=42)
    t0 = time.perf_counter()
    sp = SpectralClustering(n_clusters=3, affinity="rbf", random_state=42).fit(Xn)
    t1 = time.perf_counter()
    print(f"n={n:5d} | زمن={t1 - t0:5.2f}s | شكل affinity_matrix_: {sp.affinity_matrix_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=600, noise=0.08, random_state=42)
models = [
    ("KMeans(n_clusters=2)", lambda: KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)),
    ("Agglomerative(ward)", lambda: AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)),
    ("Agglomerative(single)", lambda: AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X)),
    ("DBSCAN(eps=0.15)", lambda: DBSCAN(eps=0.15).fit_predict(X)),
    ("Spectral(nn=20)", lambda: SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=20, random_state=42).fit_predict(X)),
    ("HDBSCAN(mcs=10)", lambda: HDBSCAN(min_cluster_size=10, copy=True).fit_predict(X)),
]
for name, fn in models:
    print(f"{name:24s} ARI={adjusted_rand_score(y, fn()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
for th in [6, 8, 10, 12, 14]:
    ag = AgglomerativeClustering(n_clusters=None, distance_threshold=float(th),
                                 linkage="ward", compute_full_tree=True).fit(Xs)
    ari = adjusted_rand_score(y, ag.labels_)
    print(f"حد={th:2d} | عناقيد={ag.n_clusters_} | ARI={ari:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [010] content/part08/ch65.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=800, n_features=40, n_informative=5,
                       random_state=42)
base = -cross_val_score(Ridge(), X, y, cv=5,
                        scoring="neg_root_mean_squared_error").mean()
pipe = make_pipeline(FeatureAgglomeration(n_clusters=20), Ridge())
agg = -cross_val_score(pipe, X, y, cv=5,
                       scoring="neg_root_mean_squared_error").mean()
print(f"Ridge مباشر (40 عموداً): RMSE={base:.3f}")
print(f"FeatureAgglo(20) + Ridge: RMSE={agg:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

datasets = {
    "الهلالان": make_moons(n_samples=600, noise=0.05, random_state=42),
    "الحلقتان": make_circles(n_samples=600, factor=0.45, noise=0.05, random_state=42),
    "الكرات": make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=42),
}
for name, (X, y) in datasets.items():
    row = [f"{name:8s}"]
    for lk in ["ward", "complete", "average", "single"]:
        lab = AgglomerativeClustering(n_clusters=2 if name != "الكرات" else 3,
                                      linkage=lk).fit_predict(X)
        row.append(f"{lk}={adjusted_rand_score(y, lab):.3f}")
    print(" | ".join(row))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_moons(n_samples=150, noise=0.05, random_state=42)
Z = linkage(X, method="ward")
print(f"شكل Z: {Z.shape} (سطر لكل اندماج من أصل {len(X) - 1})")
print(f"أول اندماج: {np.round(Z[0], 3)}")
print(f"آخر ثلاثة ارتفاعات: {np.round(Z[-3:, 2], 2)}")

R = dendrogram(Z, no_plot=True)
print(f"أوراق الدندروغرام: {len(R['leaves'])}")

for t in [2.0, 3.0, 4.0, 6.0]:
    fc = fcluster(Z, t=t, criterion="distance")
    print(f"قطع عند ارتفاع {t}: {len(np.unique(fc))} عناقيد")

ag = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
fc = fcluster(Z, t=2, criterion="maxclust")
print(f"fcluster(maxclust=2) مقابل sklearn: ARI={adjusted_rand_score(ag, fc):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=42)
for th in [5, 8, 12, 18, 25]:
    ag = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                 linkage="ward", compute_full_tree=True).fit(X)
    print(f"حد={th:2d} | عناقيد={ag.n_clusters_} | اندماجات={len(ag.distances_)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import coo_matrix
import numpy as np

rng = np.random.default_rng(42)
h, w = 40, 40
img = np.zeros((h, w))
img[:20, :20] = rng.normal(2.0, 1.2, size=(20, 20))
img[:20, 20:] = rng.normal(8.0, 1.2, size=(20, 20))
img[20:, :20] = rng.normal(8.0, 1.2, size=(20, 20))
img[20:, 20:] = rng.normal(2.0, 1.2, size=(20, 20))
y_img = np.zeros((h, w), dtype=int)
y_img[:20, 20:] = 1
y_img[20:, :20] = 2
y_img[20:, 20:] = 3
y_img = y_img.ravel()

i = np.arange(h * w)
r_i, c_i = np.divmod(i, w)
src = np.r_[i[c_i < w - 1], i[r_i < h - 1]]
dst = np.r_[i[c_i < w - 1] + 1, i[r_i < h - 1] + w]
conn = coo_matrix((np.ones(len(src)), (src, dst)), shape=(h * w, h * w)).tocsr()
conn = conn + conn.T
print(f"حواف شبكة الجيران: {conn.nnz // 2}")

X_pix = img.ravel().reshape(-1, 1)
lab_c = AgglomerativeClustering(n_clusters=4, linkage="ward",
                                connectivity=conn).fit_predict(X_pix)
lab_f = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X_pix)

def contiguity(lab):
    L = lab.reshape(h, w)
    return ((L[:, :-1] == L[:, 1:]).mean() + (L[:-1, :] == L[1:, :]).mean()) / 2

print(f"بقيد الاتصال: ARI={adjusted_rand_score(y_img, lab_c):.3f} | تماسك مكاني={contiguity(lab_c):.1%}")
print(f"بلا قيد:      ARI={adjusted_rand_score(y_img, lab_f):.3f} | تماسك مكاني={contiguity(lab_f):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3,
                                              random_state=42, stratify=yd)
clf = RidgeClassifier().fit(Xd_tr, yd_tr)
print(f"RidgeClassifier على 64 عموداً: الدقة={accuracy_score(yd_te, clf.predict(Xd_te)):.4f}")

for k in [32, 16]:
    pipe = make_pipeline(FeatureAgglomeration(n_clusters=k), RidgeClassifier())
    pipe.fit(Xd_tr, yd_tr)
    acc = accuracy_score(yd_te, pipe.predict(Xd_te))
    print(f"FeatureAgglomeration({k}) + RidgeClassifier: الدقة={acc:.4f}")

fa = FeatureAgglomeration(n_clusters=4).fit(Xd_tr)
print(f"مجموعات الأعمدة (أول 20 بكسل): {fa.labels_[:20]}")
Xt = np.asarray(fa.transform(Xd_tr))
print(f"شكل التحويل: {Xt.shape} | العودة: {fa.inverse_transform(Xt).shape}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

Xm, ym = make_moons(n_samples=600, noise=0.05, random_state=42)
Xc, yc = make_circles(n_samples=600, factor=0.45, noise=0.05, random_state=42)
lab_m = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                           n_neighbors=20, random_state=42).fit_predict(Xm)
lab_c = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                           n_neighbors=40, random_state=42).fit_predict(Xc)
print(f"الهلالان: ARI={adjusted_rand_score(ym, lab_m):.4f}")
print(f"الحلقتان: ARI={adjusted_rand_score(yc, lab_c):.4f}")

for al in ["kmeans", "discretize", "cluster_qr"]:
    lab = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                             n_neighbors=20, assign_labels=al,
                             random_state=42).fit_predict(Xm)
    print(f"assign_labels={al:10s}: ARI={adjusted_rand_score(ym, lab):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
import time

for n in [1200, 3600]:
    Xn, _ = make_blobs(n_samples=n, centers=3, random_state=42)
    t0 = time.perf_counter()
    sp = SpectralClustering(n_clusters=3, affinity="rbf", random_state=42).fit(Xn)
    t1 = time.perf_counter()
    print(f"n={n:5d} | زمن={t1 - t0:5.2f}s | شكل affinity_matrix_: {sp.affinity_matrix_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=600, noise=0.08, random_state=42)
models = [
    ("KMeans(n_clusters=2)", lambda: KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)),
    ("Agglomerative(ward)", lambda: AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)),
    ("Agglomerative(single)", lambda: AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X)),
    ("DBSCAN(eps=0.15)", lambda: DBSCAN(eps=0.15).fit_predict(X)),
    ("Spectral(nn=20)", lambda: SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=20, random_state=42).fit_predict(X)),
    ("HDBSCAN(mcs=10)", lambda: HDBSCAN(min_cluster_size=10, copy=True).fit_predict(X)),
]
for name, fn in models:
    print(f"{name:24s} ARI={adjusted_rand_score(y, fn()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
for th in [6, 8, 10, 12, 14]:
    ag = AgglomerativeClustering(n_clusters=None, distance_threshold=float(th),
                                 linkage="ward", compute_full_tree=True).fit(Xs)
    ari = adjusted_rand_score(y, ag.labels_)
    print(f"حد={th:2d} | عناقيد={ag.n_clusters_} | ARI={ari:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-65-ch65.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=800, n_features=40, n_informative=5,
                       random_state=42)
base = -cross_val_score(Ridge(), X, y, cv=5,
                        scoring="neg_root_mean_squared_error").mean()
pipe = make_pipeline(FeatureAgglomeration(n_clusters=20), Ridge())
agg = -cross_val_score(pipe, X, y, cv=5,
                       scoring="neg_root_mean_squared_error").mean()
print(f"Ridge مباشر (40 عموداً): RMSE={base:.3f}")
print(f"FeatureAgglo(20) + Ridge: RMSE={agg:.3f}")
