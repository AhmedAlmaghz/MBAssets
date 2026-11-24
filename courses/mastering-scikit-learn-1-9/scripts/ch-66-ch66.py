#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كيف تقيس نجاح تجميعٍ بلا إجابة؟
Mastering scikit-learn 1.9 — Chapter 66 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-66-ch66 (lesson 66)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-66-ch66.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كيف تقيس نجاح تجميعٍ بلا إجابة؟
Mastering scikit-learn 1.9 — Chapter 66 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-66-ch66 (lesson 66)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

for std in [0.6, 1.5, 2.6]:
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=std, random_state=42)
    s = silhouette_score(X, y)                      # متوسط واحد للكل
    samples = silhouette_samples(X, y)              # قيمة لكل نقطة
    per_cluster = np.array([samples[y == c].mean() for c in range(3)])
    print(f"cluster_std={std} | score={s:.4f} | متوسطات العناقيد: "
          f"{np.round(per_cluster, 3)} | نقاط سالبة: {(samples < 0).sum()}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

X, y = make_blobs(n_samples=600, centers=3, cluster_std=2.6, random_state=42)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
samples = silhouette_samples(X, km.labels_)

print(f"silhouette_score الإجمالي: {silhouette_score(X, km.labels_):.4f}")
print(f"تحقق هوية مهم: متوسط العيّنات = {samples.mean():.4f}")
for c in range(3):
    vals = samples[km.labels_ == c]
    print(f"عنقود {c}: متوسط الظل {vals.mean():+.4f} | نقاط سالبة: {(vals < 0).sum()} من {len(vals)}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

X, y_true = make_moons(n_samples=600, noise=0.06, random_state=42)
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_km = km.fit_predict(X)

print(f"silhouette للتقسيم الحقيقي (الهلالان): {silhouette_score(X, y_true):.4f}")
print(f"silhouette لتقسيم K-Means الشاقولي:    {silhouette_score(X, labels_km):.4f}")
print(f"ARI لتقسيم K-Means مقابل الحقيقة:      {adjusted_rand_score(y_true, labels_km):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=42)
print(f"{'k':>3} | {'Silhouette':>10} | {'Davies-Bouldin':>14} | {'Calinski-Harabasz':>17}")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"{k:3d} | {silhouette_score(X, labels):10.4f} | "
          f"{davies_bouldin_score(X, labels):14.4f} | "
          f"{calinski_harabasz_score(X, labels):17.1f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=42)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

# تدوير أسماء العنقود: 0 يصير 1، و 1 يصير 2، و 2 يصير 0 — البنية لم تتغير إطلاقاً!
rotated = (labels + 1) % 3

print(f"تطابق مباشر (==) قبل التدوير: {(y_true == labels).mean():.2%}")
print(f"تطابق مباشر (==) بعد التدوير: {(y_true == rotated).mean():.2%}")
print(f"ARI قبل التدوير:  {adjusted_rand_score(y_true, labels):.4f}")
print(f"ARI بعد التدوير:  {adjusted_rand_score(y_true, rotated):.4f}")
print(f"NMI بعد التدوير:  {normalized_mutual_info_score(y_true, rotated):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             fowlkes_mallows_score)
import pandas as pd

X, y_true = make_blobs(n_samples=900,
                       centers=[[0, 0], [5, 0], [6.8, 0]],
                       cluster_std=1.0, random_state=42)

rows = []
for name, k in [("K-Means k=2", 2), ("K-Means k=3", 3), ("K-Means k=4", 4)]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    rows.append({
        "التقسيم": name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "V-measure": v_measure_score(y_true, labels),
        "FMI": fowlkes_mallows_score(y_true, labels),
    })

df = pd.DataFrame(rows).set_index("التقسيم").round(3)
df

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
import pandas as pd

X, y_true = load_iris(return_X_y=True)

rows = []
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    rows.append({
        "k": k,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "ARI": adjusted_rand_score(y_true, labels),
    })

df = pd.DataFrame(rows).set_index("k").round(3)
best = df.idxmax()[["Silhouette", "Calinski-Harabasz", "ARI"]]
best["Davies-Bouldin"] = df["Davies-Bouldin"].idxmin()
print(f"اختيار كل مقياس لقيمة k المثلى: {dict(best)}")
df

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch66.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

X, y_true = make_blobs(n_samples=900, centers=3, cluster_std=1.3, random_state=42)

for k in [3, 6, 12]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"k={k:2d} | ARI = {adjusted_rand_score(y_true, labels):.4f} | "
          f"NMI = {normalized_mutual_info_score(y_true, labels):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

for std in [0.6, 1.5, 2.6]:
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=std, random_state=42)
    s = silhouette_score(X, y)                      # متوسط واحد للكل
    samples = silhouette_samples(X, y)              # قيمة لكل نقطة
    per_cluster = np.array([samples[y == c].mean() for c in range(3)])
    print(f"cluster_std={std} | score={s:.4f} | متوسطات العناقيد: "
          f"{np.round(per_cluster, 3)} | نقاط سالبة: {(samples < 0).sum()}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

X, y = make_blobs(n_samples=600, centers=3, cluster_std=2.6, random_state=42)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
samples = silhouette_samples(X, km.labels_)

print(f"silhouette_score الإجمالي: {silhouette_score(X, km.labels_):.4f}")
print(f"تحقق هوية مهم: متوسط العيّنات = {samples.mean():.4f}")
for c in range(3):
    vals = samples[km.labels_ == c]
    print(f"عنقود {c}: متوسط الظل {vals.mean():+.4f} | نقاط سالبة: {(vals < 0).sum()} من {len(vals)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

X, y_true = make_moons(n_samples=600, noise=0.06, random_state=42)
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_km = km.fit_predict(X)

print(f"silhouette للتقسيم الحقيقي (الهلالان): {silhouette_score(X, y_true):.4f}")
print(f"silhouette لتقسيم K-Means الشاقولي:    {silhouette_score(X, labels_km):.4f}")
print(f"ARI لتقسيم K-Means مقابل الحقيقة:      {adjusted_rand_score(y_true, labels_km):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=42)
print(f"{'k':>3} | {'Silhouette':>10} | {'Davies-Bouldin':>14} | {'Calinski-Harabasz':>17}")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"{k:3d} | {silhouette_score(X, labels):10.4f} | "
          f"{davies_bouldin_score(X, labels):14.4f} | "
          f"{calinski_harabasz_score(X, labels):17.1f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=42)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

# تدوير أسماء العنقود: 0 يصير 1، و 1 يصير 2، و 2 يصير 0 — البنية لم تتغير إطلاقاً!
rotated = (labels + 1) % 3

print(f"تطابق مباشر (==) قبل التدوير: {(y_true == labels).mean():.2%}")
print(f"تطابق مباشر (==) بعد التدوير: {(y_true == rotated).mean():.2%}")
print(f"ARI قبل التدوير:  {adjusted_rand_score(y_true, labels):.4f}")
print(f"ARI بعد التدوير:  {adjusted_rand_score(y_true, rotated):.4f}")
print(f"NMI بعد التدوير:  {normalized_mutual_info_score(y_true, rotated):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,
                             fowlkes_mallows_score)
import pandas as pd

X, y_true = make_blobs(n_samples=900,
                       centers=[[0, 0], [5, 0], [6.8, 0]],
                       cluster_std=1.0, random_state=42)

rows = []
for name, k in [("K-Means k=2", 2), ("K-Means k=3", 3), ("K-Means k=4", 4)]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    rows.append({
        "التقسيم": name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "V-measure": v_measure_score(y_true, labels),
        "FMI": fowlkes_mallows_score(y_true, labels),
    })

df = pd.DataFrame(rows).set_index("التقسيم").round(3)
df

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
import pandas as pd

X, y_true = load_iris(return_X_y=True)

rows = []
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    rows.append({
        "k": k,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "ARI": adjusted_rand_score(y_true, labels),
    })

df = pd.DataFrame(rows).set_index("k").round(3)
best = df.idxmax()[["Silhouette", "Calinski-Harabasz", "ARI"]]
best["Davies-Bouldin"] = df["Davies-Bouldin"].idxmin()
print(f"اختيار كل مقياس لقيمة k المثلى: {dict(best)}")
df

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-66-ch66.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

X, y_true = make_blobs(n_samples=900, centers=3, cluster_std=1.3, random_state=42)

for k in [3, 6, 12]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"k={k:2d} | ARI = {adjusted_rand_score(y_true, labels):.4f} | "
          f"NMI = {normalized_mutual_info_score(y_true, labels):.4f}")
