#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — HDBSCAN: عدد كتل غير معروف مسبقاً
Mastering scikit-learn 1.9 — HDBSCAN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-64-ch64 (lesson 64)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-64-ch64.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — HDBSCAN: عدد كتل غير معروف مسبقاً
Mastering scikit-learn 1.9 — HDBSCAN

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-64-ch64 (lesson 64)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
import numpy as np

hd = HDBSCAN(copy=True)
print("افتراضيات 1.9:", {k: hd.get_params()[k] for k in
      ["min_cluster_size", "min_samples", "cluster_selection_epsilon",
       "cluster_selection_method", "metric"]})

X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
hd.fit(X)
lab = hd.labels_
print(f"عناقيد مكتشفة: {np.unique(lab).tolist()}")
print(f"مدى probabilities_: {hd.probabilities_.min():.3f} إلى {hd.probabilities_.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, HDBSCAN
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

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
lab = hd.labels_
k = len(set(lab)) - (1 if -1 in lab else 0)
print(f"HDBSCAN (بلا eps): عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

for mcs in [5, 15, 30, 60]:
    hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"min_cluster_size={mcs:3d}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
import numpy as np

for ms in [None, 5, 15, 30]:
    hd = HDBSCAN(min_cluster_size=15, min_samples=ms, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    ms_txt = "None" if ms is None else str(ms)
    print(f"min_samples={ms_txt:>5}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

for method in ["eom", "leaf"]:
    hd = HDBSCAN(min_cluster_size=15, cluster_selection_method=method, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"method={method:6s}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
hd = HDBSCAN(min_cluster_size=15, cluster_selection_epsilon=0.5, copy=True)
# كل عناقيد أقرب من 0.5 في الشجرة المكثفة تُدمج قبل الإخراج

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
import numpy as np

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
lab = hd.labels_
for c in sorted(set(lab)):
    sel = lab == c
    p = hd.probabilities_[sel]
    if c == -1:
        print(f"ضجيج (-1): n={int(sel.sum())} | probabilities من {p.min():.3f} إلى {p.max():.3f}")
    else:
        print(f"عنقود {c}: n={int(sel.sum())} | متوسط={p.mean():.3f}"
              f" | أدنى={p.min():.3f} | أقل من 0.5: {(p < 0.5).mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

print("DBSCAN عبر مساحة eps:")
for eps in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
    lab = DBSCAN(eps=eps, min_samples=8).fit(X_dual).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"  eps={eps:.1f}: عناقيد={k} | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
print(f"HDBSCAN (بلا eps): ARI={adjusted_rand_score(y_dual[mask], hd.labels_[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

Xd, yd = load_digits(return_X_y=True)
hd_raw = HDBSCAN(min_cluster_size=15, copy=True).fit(Xd)
lab_raw = hd_raw.labels_
k_raw = len(set(lab_raw)) - (1 if -1 in lab_raw else 0)
print(f"خام (64 بعداً): عناقيد={k_raw} | ضجيج={(lab_raw == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(yd, lab_raw):.3f}")

Xp = PCA(n_components=0.90, random_state=42).fit_transform(Xd)
hd_pca = HDBSCAN(min_cluster_size=15, copy=True).fit(Xp)
lab_pca = hd_pca.labels_
k_pca = len(set(lab_pca)) - (1 if -1 in lab_pca else 0)
print(f"PCA ({Xp.shape[1]} بعداً): عناقيد={k_pca} | ضجيج={(lab_pca == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(yd, lab_pca):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [010] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=[400, 300, 60], cluster_std=[0.5, 0.8, 1.2],
                  random_state=42)
rng = np.random.default_rng(42)
X3 = np.vstack([X, rng.uniform(-12, 12, size=(40, 2))])
y3 = np.concatenate([y, np.full(40, -1)])
mask = y3 != -1

for mcs in [10, 25, 50]:
    hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X3)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"mcs={mcs:3d}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y3[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [011] content/part08/ch64.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_moons(n_samples=600, noise=0.05, random_state=42)
for method in ["eom", "leaf"]:
    hd = HDBSCAN(min_cluster_size=15, cluster_selection_method=method,
                 copy=True).fit(X)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"{method:5s}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y, lab):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
import numpy as np

hd = HDBSCAN(copy=True)
print("افتراضيات 1.9:", {k: hd.get_params()[k] for k in
      ["min_cluster_size", "min_samples", "cluster_selection_epsilon",
       "cluster_selection_method", "metric"]})

X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
hd.fit(X)
lab = hd.labels_
print(f"عناقيد مكتشفة: {np.unique(lab).tolist()}")
print(f"مدى probabilities_: {hd.probabilities_.min():.3f} إلى {hd.probabilities_.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, HDBSCAN
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

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
lab = hd.labels_
k = len(set(lab)) - (1 if -1 in lab else 0)
print(f"HDBSCAN (بلا eps): عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

for mcs in [5, 15, 30, 60]:
    hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"min_cluster_size={mcs:3d}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
import numpy as np

for ms in [None, 5, 15, 30]:
    hd = HDBSCAN(min_cluster_size=15, min_samples=ms, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    ms_txt = "None" if ms is None else str(ms)
    print(f"min_samples={ms_txt:>5}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

for method in ["eom", "leaf"]:
    hd = HDBSCAN(min_cluster_size=15, cluster_selection_method=method, copy=True).fit(X_dual)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"method={method:6s}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
hd = HDBSCAN(min_cluster_size=15, cluster_selection_epsilon=0.5, copy=True)
# كل عناقيد أقرب من 0.5 في الشجرة المكثفة تُدمج قبل الإخراج

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import HDBSCAN
import numpy as np

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
lab = hd.labels_
for c in sorted(set(lab)):
    sel = lab == c
    p = hd.probabilities_[sel]
    if c == -1:
        print(f"ضجيج (-1): n={int(sel.sum())} | probabilities من {p.min():.3f} إلى {p.max():.3f}")
    else:
        print(f"عنقود {c}: n={int(sel.sum())} | متوسط={p.mean():.3f}"
              f" | أدنى={p.min():.3f} | أقل من 0.5: {(p < 0.5).mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

print("DBSCAN عبر مساحة eps:")
for eps in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
    lab = DBSCAN(eps=eps, min_samples=8).fit(X_dual).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"  eps={eps:.1f}: عناقيد={k} | ARI={adjusted_rand_score(y_dual[mask], lab[mask]):.3f}")

hd = HDBSCAN(min_cluster_size=15, copy=True).fit(X_dual)
print(f"HDBSCAN (بلا eps): ARI={adjusted_rand_score(y_dual[mask], hd.labels_[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

Xd, yd = load_digits(return_X_y=True)
hd_raw = HDBSCAN(min_cluster_size=15, copy=True).fit(Xd)
lab_raw = hd_raw.labels_
k_raw = len(set(lab_raw)) - (1 if -1 in lab_raw else 0)
print(f"خام (64 بعداً): عناقيد={k_raw} | ضجيج={(lab_raw == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(yd, lab_raw):.3f}")

Xp = PCA(n_components=0.90, random_state=42).fit_transform(Xd)
hd_pca = HDBSCAN(min_cluster_size=15, copy=True).fit(Xp)
lab_pca = hd_pca.labels_
k_pca = len(set(lab_pca)) - (1 if -1 in lab_pca else 0)
print(f"PCA ({Xp.shape[1]} بعداً): عناقيد={k_pca} | ضجيج={(lab_pca == -1).mean():.1%}"
      f" | ARI={adjusted_rand_score(yd, lab_pca):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_blobs(n_samples=[400, 300, 60], cluster_std=[0.5, 0.8, 1.2],
                  random_state=42)
rng = np.random.default_rng(42)
X3 = np.vstack([X, rng.uniform(-12, 12, size=(40, 2))])
y3 = np.concatenate([y, np.full(40, -1)])
mask = y3 != -1

for mcs in [10, 25, 50]:
    hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X3)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"mcs={mcs:3d}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y3[mask], lab[mask]):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-64-ch64.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y = make_moons(n_samples=600, noise=0.05, random_state=42)
for method in ["eom", "leaf"]:
    hd = HDBSCAN(min_cluster_size=15, cluster_selection_method=method,
                 copy=True).fit(X)
    lab = hd.labels_
    k = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"{method:5s}: عناقيد={k} | ضجيج={(lab == -1).mean():.1%}"
          f" | ARI={adjusted_rand_score(y, lab):.3f}")
