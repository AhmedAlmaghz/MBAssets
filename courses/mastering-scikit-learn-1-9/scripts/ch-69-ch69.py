#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقليل الأبعاد غير الخطي
Mastering scikit-learn 1.9 — Chapter 69 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-69-ch69 (lesson 69)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-69-ch69.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقليل الأبعاد غير الخطي
Mastering scikit-learn 1.9 — Chapter 69 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-69-ch69 (lesson 69)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import numpy as np

X, t = make_swiss_roll(n_samples=500, noise=0.10, random_state=42)
Z = np.asarray(PCA(n_components=2, random_state=42).fit_transform(X))

# كم من الجوار الأصلي نجا بعد الإسقاط الخطي؟
trust = trustworthiness(X, Z, n_neighbors=8)

# ابحث عن أكثر زوج خُدع: بعيد جداً على الورقة، متلاصق في الإسقاط
n = len(X)
D3 = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
D2 = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
iu = np.triu_indices(n, k=1)
scam = (D3[iu] - D2[iu]).argmax()
i, j = iu[0][scam], iu[1][scam]

print(f"موثوقية إسقاط PCA على اللفافة: {trust:.4f}")
print(f"أشد زوج خُدع: مسافته على اللفافة = {D3[i, j]:.2f}, وبعد الإسقاط = {D2[i, j]:.2f}")

Xs, _ = make_s_curve(n_samples=500, noise=0.08, random_state=42)
Zs = np.asarray(PCA(n_components=2, random_state=42).fit_transform(Xs))
print(f"موثوقية PCA على منحنى S: {trustworthiness(Xs, Zs, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, trustworthiness
import numpy as np
import time

methods_global = {
    "PCA": PCA(n_components=2, random_state=42),
    "KernelPCA": KernelPCA(n_components=2, kernel="rbf", gamma=0.03, random_state=42),
    "MDS": MDS(n_components=2, n_init=1, max_iter=100, init="classical_mds", random_state=42),
}
for name, m in methods_global.items():
    t0 = time.perf_counter()
    Z = np.asarray(m.fit_transform(X))
    dt = (time.perf_counter() - t0) * 1000
    print(f"{name:12s} | زمن: {dt:7.1f} ms | موثوقية: {trustworthiness(X, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE, trustworthiness
import numpy as np
import time

methods_local = {
    "Isomap": Isomap(n_neighbors=12, n_components=2, n_jobs=-1),
    "LLE": LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=42),
    "SpectralEmbedding": SpectralEmbedding(n_neighbors=12, n_components=2, random_state=42),
    "TSNE": TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42, n_jobs=-1),
}
for name, m in methods_local.items():
    t0 = time.perf_counter()
    Z = np.asarray(m.fit_transform(X))
    dt = (time.perf_counter() - t0) * 1000
    print(f"{name:20s} | زمن: {dt:7.1f} ms | موثوقية: {trustworthiness(X, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE, trustworthiness
import numpy as np
import time

Xs, _ = make_swiss_roll(n_samples=300, noise=0.10, random_state=42)
for p in (5, 30, 100):
    t0 = time.perf_counter()
    Z = np.asarray(TSNE(n_components=2, perplexity=p, init="pca",
                        learning_rate="auto", random_state=42).fit_transform(Xs))
    dt = (time.perf_counter() - t0) * 1000
    tr = trustworthiness(Xs, Z, n_neighbors=8)
    print(f"perplexity={p:3d} | زمن: {dt:6.1f} ms | موثوقية: {tr:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE, Isomap
import numpy as np

sub = X[:150]
tsne = TSNE(n_components=2, perplexity=20, init="pca", learning_rate="auto", random_state=42).fit(sub)
iso = Isomap(n_neighbors=10, n_components=2).fit(sub)

try:
    tsne.transform(sub[:3])
except AttributeError as err:
    print(f"TSNE: رفض صريح — {err}")

Z_new = np.asarray(iso.transform(sub[:3]))
print(f"Isomap: transform يعمل بأمان، خرج بصيغة {Z_new.shape}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import Isomap, trustworthiness
import numpy as np

for k in (5, 10, 15, 25, 40):
    Z = np.asarray(Isomap(n_neighbors=k, n_components=2).fit_transform(X))
    tr = trustworthiness(X, Z, n_neighbors=8)
    print(f"n_neighbors={k:2d} | موثوقية: {tr:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.manifold import trustworthiness
import numpy as np

Xr, _ = make_swiss_roll(n_samples=500, noise=0.10, random_state=42)
for g in (0.001, 0.01, 0.1, 1.0):
    Z = np.asarray(KernelPCA(n_components=2, kernel="rbf", gamma=g, random_state=42).fit_transform(Xr))
    print(f"gamma={g:<5} | موثوقية: {trustworthiness(Xr, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch69.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap, trustworthiness
import numpy as np

Xs, _ = make_s_curve(n_samples=400, noise=0.08, random_state=42)
for k in (4, 8, 15, 30):
    Z = np.asarray(Isomap(n_neighbors=k, n_components=2).fit_transform(Xs))
    print(f"n_neighbors={k:2d} | موثوقية: {trustworthiness(Xs, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import numpy as np

X, t = make_swiss_roll(n_samples=500, noise=0.10, random_state=42)
Z = np.asarray(PCA(n_components=2, random_state=42).fit_transform(X))

# كم من الجوار الأصلي نجا بعد الإسقاط الخطي؟
trust = trustworthiness(X, Z, n_neighbors=8)

# ابحث عن أكثر زوج خُدع: بعيد جداً على الورقة، متلاصق في الإسقاط
n = len(X)
D3 = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
D2 = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
iu = np.triu_indices(n, k=1)
scam = (D3[iu] - D2[iu]).argmax()
i, j = iu[0][scam], iu[1][scam]

print(f"موثوقية إسقاط PCA على اللفافة: {trust:.4f}")
print(f"أشد زوج خُدع: مسافته على اللفافة = {D3[i, j]:.2f}, وبعد الإسقاط = {D2[i, j]:.2f}")

Xs, _ = make_s_curve(n_samples=500, noise=0.08, random_state=42)
Zs = np.asarray(PCA(n_components=2, random_state=42).fit_transform(Xs))
print(f"موثوقية PCA على منحنى S: {trustworthiness(Xs, Zs, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, trustworthiness
import numpy as np
import time

methods_global = {
    "PCA": PCA(n_components=2, random_state=42),
    "KernelPCA": KernelPCA(n_components=2, kernel="rbf", gamma=0.03, random_state=42),
    "MDS": MDS(n_components=2, n_init=1, max_iter=100, init="classical_mds", random_state=42),
}
for name, m in methods_global.items():
    t0 = time.perf_counter()
    Z = np.asarray(m.fit_transform(X))
    dt = (time.perf_counter() - t0) * 1000
    print(f"{name:12s} | زمن: {dt:7.1f} ms | موثوقية: {trustworthiness(X, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE, trustworthiness
import numpy as np
import time

methods_local = {
    "Isomap": Isomap(n_neighbors=12, n_components=2, n_jobs=-1),
    "LLE": LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=42),
    "SpectralEmbedding": SpectralEmbedding(n_neighbors=12, n_components=2, random_state=42),
    "TSNE": TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42, n_jobs=-1),
}
for name, m in methods_local.items():
    t0 = time.perf_counter()
    Z = np.asarray(m.fit_transform(X))
    dt = (time.perf_counter() - t0) * 1000
    print(f"{name:20s} | زمن: {dt:7.1f} ms | موثوقية: {trustworthiness(X, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE, trustworthiness
import numpy as np
import time

Xs, _ = make_swiss_roll(n_samples=300, noise=0.10, random_state=42)
for p in (5, 30, 100):
    t0 = time.perf_counter()
    Z = np.asarray(TSNE(n_components=2, perplexity=p, init="pca",
                        learning_rate="auto", random_state=42).fit_transform(Xs))
    dt = (time.perf_counter() - t0) * 1000
    tr = trustworthiness(Xs, Z, n_neighbors=8)
    print(f"perplexity={p:3d} | زمن: {dt:6.1f} ms | موثوقية: {tr:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE, Isomap
import numpy as np

sub = X[:150]
tsne = TSNE(n_components=2, perplexity=20, init="pca", learning_rate="auto", random_state=42).fit(sub)
iso = Isomap(n_neighbors=10, n_components=2).fit(sub)

try:
    tsne.transform(sub[:3])
except AttributeError as err:
    print(f"TSNE: رفض صريح — {err}")

Z_new = np.asarray(iso.transform(sub[:3]))
print(f"Isomap: transform يعمل بأمان، خرج بصيغة {Z_new.shape}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.manifold import Isomap, trustworthiness
import numpy as np

for k in (5, 10, 15, 25, 40):
    Z = np.asarray(Isomap(n_neighbors=k, n_components=2).fit_transform(X))
    tr = trustworthiness(X, Z, n_neighbors=8)
    print(f"n_neighbors={k:2d} | موثوقية: {tr:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.manifold import trustworthiness
import numpy as np

Xr, _ = make_swiss_roll(n_samples=500, noise=0.10, random_state=42)
for g in (0.001, 0.01, 0.1, 1.0):
    Z = np.asarray(KernelPCA(n_components=2, kernel="rbf", gamma=g, random_state=42).fit_transform(Xr))
    print(f"gamma={g:<5} | موثوقية: {trustworthiness(Xr, Z, n_neighbors=8):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-69-ch69.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap, trustworthiness
import numpy as np

Xs, _ = make_s_curve(n_samples=400, noise=0.08, random_state=42)
for k in (4, 8, 15, 30):
    Z = np.asarray(Isomap(n_neighbors=k, n_components=2).fit_transform(Xs))
    print(f"n_neighbors={k:2d} | موثوقية: {trustworthiness(Xs, Z, n_neighbors=8):.4f}")
