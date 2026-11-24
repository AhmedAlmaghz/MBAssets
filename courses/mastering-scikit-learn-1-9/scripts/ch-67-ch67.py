#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النماذج المزيجية من الغاussians
Mastering scikit-learn 1.9 — ussians

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-67-ch67 (lesson 67)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-67-ch67.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النماذج المزيجية من الغاussians
Mastering scikit-learn 1.9 — ussians

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-67-ch67 (lesson 67)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd

X_raw, y_true = make_blobs(n_samples=900, centers=[[0, 0], [9, 0], [12, 0]],
                           cluster_std=[3.0, 0.7, 0.7], random_state=42)
A = np.array([[0.6, -0.65], [-0.35, 0.9]])   # إطالة وميلان للفضاء كله
X = X_raw @ A

rows = []
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
rows.append({"النموذج": "K-Means (صلب)",
             "ARI": f"{adjusted_rand_score(y_true, km.labels_):.3f}",
             "متوسط اللوغاريتم": "—"})
for ct in ["spherical", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X)
    rows.append({"النموذج": f"GMM ({ct})",
                 "ARI": f"{adjusted_rand_score(y_true, gm.predict(X)):.3f}",
                 "متوسط اللوغاريتم": f"{gm.score(X):.3f}"})

pd.DataFrame(rows).set_index("النموذج")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
# X و y_true من القسم ٨.١
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

rows = []
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X)
    rows.append({
        "covariance_type": ct,
        "ARI": adjusted_rand_score(y_true, gm.predict(X)),
        "متوسط اللوغاريتم": round(gm.score(X), 3),
        "عدد البارامترات": gm._n_parameters(),
    })

df = pd.DataFrame(rows).set_index("covariance_type")
df

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

print(f"{'k':>2} | {'BIC':>9} | {'AIC':>9}")
bics, aics = [], []
for k in range(1, 9):
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=3, random_state=42).fit(X)
    bics.append(gm.bic(X))
    aics.append(gm.aic(X))
    print(f"{k:2d} | {gm.bic(X):9.1f} | {gm.aic(X):9.1f}")

print(f"أدنى BIC عند k = {int(np.argmin(bics)) + 1} | أدنى AIC عند k = {int(np.argmin(aics)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, covariance_type="full",
                     n_init=3, random_state=42).fit(X)

# نقطتان على منتصف المسافة بين مركزين: من هما؟
for i, j in [(0, 1), (0, 2)]:
    mid = ((gm.means_[i] + gm.means_[j]) / 2).reshape(1, -1)
    print(f"منتصف المسافة بين المكوّنين {i} و {j}: مسؤوليات {gm.predict_proba(mid).round(3)}")

# الكثافة: عينات عادية مقابل نقطة بعيدة عن كل شيء
log_dens = gm.score_samples(X[:3])
far = np.array([[16.0, -14.0]])
print(f"كثافة لوغاريتمية لعينات عادية: {np.round(log_dens, 2)}")
print(f"كثافة لوغاريتمية لنقطة بعيدة {far.tolist()}: {gm.score_samples(far).round(2)}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_small, _, y_small, _ = train_test_split(
    X, y_true, train_size=45, random_state=42, stratify=y_true)

def ll(model, data):
    return float(model.score(data) * len(data))

print(f"{'الإعداد':32} | {'تدريب':>9} | {'كامل البيانات':>13}")
for reg in [1e-6, 0.1]:
    gm_over = GaussianMixture(n_components=8, covariance_type="full",
                              reg_covar=reg, random_state=42).fit(X_small)
    print(f"full و k=8 و reg_covar={reg:<6} | {ll(gm_over, X_small):9.1f} | "
          f"{ll(gm_over, X):13.1f}")

gm_right = GaussianMixture(n_components=3, covariance_type="full",
                           random_state=42).fit(X_small)
print(f"{'full و k=3 و reg_covar=1e-06':32} | {ll(gm_right, X_small):9.1f} | "
      f"{ll(gm_right, X):13.1f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(
    n_components=10, covariance_type="full",
    weight_concentration_prior=0.1, max_iter=400, random_state=42).fit(X)

weights = np.sort(bgm.weights_)[::-1]
print(f"الأوزان مرتبة تنازلياً: {np.round(weights, 4)}")
print(f"مكوّنات فعالة (وزن أكبر من 0.05): {(bgm.weights_ > 0.05).sum()}")
print(f"وزن أقوى مكوّن: {bgm.weights_.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

X_iris, y_iris = load_iris(return_X_y=True)

rows = []
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X_iris)
    rows.append({
        "covariance_type": ct,
        "ARI": adjusted_rand_score(y_iris, gm.predict(X_iris)),
        "عدد البارامترات": gm._n_parameters(),
    })

df = pd.DataFrame(rows).set_index("covariance_type")
df

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch67.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, covariance_type="full",
                     n_init=3, random_state=42).fit(X)

log_dens = gm.score_samples(X)
threshold = np.percentile(log_dens, 2)      # عتبة المئين الثاني
print(f"عتبة الكثافة (المئين 2): {threshold:.2f}")

outliers = np.array([[16.0, -14.0], [-17.0, 12.0], [14.0, 15.0],
                     [-15.0, -16.0], [18.0, 4.0]])
caught = (gm.score_samples(outliers) < threshold).sum()
print(f"نقاط شاذة محقونة: {len(outliers)} | الملتقطة تحت العتبة: {caught}")
print(f"نسبة الالتقاط: {caught / len(outliers):.0%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd

X_raw, y_true = make_blobs(n_samples=900, centers=[[0, 0], [9, 0], [12, 0]],
                           cluster_std=[3.0, 0.7, 0.7], random_state=42)
A = np.array([[0.6, -0.65], [-0.35, 0.9]])   # إطالة وميلان للفضاء كله
X = X_raw @ A

rows = []
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
rows.append({"النموذج": "K-Means (صلب)",
             "ARI": f"{adjusted_rand_score(y_true, km.labels_):.3f}",
             "متوسط اللوغاريتم": "—"})
for ct in ["spherical", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X)
    rows.append({"النموذج": f"GMM ({ct})",
                 "ARI": f"{adjusted_rand_score(y_true, gm.predict(X)):.3f}",
                 "متوسط اللوغاريتم": f"{gm.score(X):.3f}"})

pd.DataFrame(rows).set_index("النموذج")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# X و y_true من القسم ٨.١
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

rows = []
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X)
    rows.append({
        "covariance_type": ct,
        "ARI": adjusted_rand_score(y_true, gm.predict(X)),
        "متوسط اللوغاريتم": round(gm.score(X), 3),
        "عدد البارامترات": gm._n_parameters(),
    })

df = pd.DataFrame(rows).set_index("covariance_type")
df

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

print(f"{'k':>2} | {'BIC':>9} | {'AIC':>9}")
bics, aics = [], []
for k in range(1, 9):
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=3, random_state=42).fit(X)
    bics.append(gm.bic(X))
    aics.append(gm.aic(X))
    print(f"{k:2d} | {gm.bic(X):9.1f} | {gm.aic(X):9.1f}")

print(f"أدنى BIC عند k = {int(np.argmin(bics)) + 1} | أدنى AIC عند k = {int(np.argmin(aics)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, covariance_type="full",
                     n_init=3, random_state=42).fit(X)

# نقطتان على منتصف المسافة بين مركزين: من هما؟
for i, j in [(0, 1), (0, 2)]:
    mid = ((gm.means_[i] + gm.means_[j]) / 2).reshape(1, -1)
    print(f"منتصف المسافة بين المكوّنين {i} و {j}: مسؤوليات {gm.predict_proba(mid).round(3)}")

# الكثافة: عينات عادية مقابل نقطة بعيدة عن كل شيء
log_dens = gm.score_samples(X[:3])
far = np.array([[16.0, -14.0]])
print(f"كثافة لوغاريتمية لعينات عادية: {np.round(log_dens, 2)}")
print(f"كثافة لوغاريتمية لنقطة بعيدة {far.tolist()}: {gm.score_samples(far).round(2)}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_small, _, y_small, _ = train_test_split(
    X, y_true, train_size=45, random_state=42, stratify=y_true)

def ll(model, data):
    return float(model.score(data) * len(data))

print(f"{'الإعداد':32} | {'تدريب':>9} | {'كامل البيانات':>13}")
for reg in [1e-6, 0.1]:
    gm_over = GaussianMixture(n_components=8, covariance_type="full",
                              reg_covar=reg, random_state=42).fit(X_small)
    print(f"full و k=8 و reg_covar={reg:<6} | {ll(gm_over, X_small):9.1f} | "
          f"{ll(gm_over, X):13.1f}")

gm_right = GaussianMixture(n_components=3, covariance_type="full",
                           random_state=42).fit(X_small)
print(f"{'full و k=3 و reg_covar=1e-06':32} | {ll(gm_right, X_small):9.1f} | "
      f"{ll(gm_right, X):13.1f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(
    n_components=10, covariance_type="full",
    weight_concentration_prior=0.1, max_iter=400, random_state=42).fit(X)

weights = np.sort(bgm.weights_)[::-1]
print(f"الأوزان مرتبة تنازلياً: {np.round(weights, 4)}")
print(f"مكوّنات فعالة (وزن أكبر من 0.05): {(bgm.weights_ > 0.05).sum()}")
print(f"وزن أقوى مكوّن: {bgm.weights_.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

X_iris, y_iris = load_iris(return_X_y=True)

rows = []
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct,
                         n_init=5, random_state=42).fit(X_iris)
    rows.append({
        "covariance_type": ct,
        "ARI": adjusted_rand_score(y_iris, gm.predict(X_iris)),
        "عدد البارامترات": gm._n_parameters(),
    })

df = pd.DataFrame(rows).set_index("covariance_type")
df

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-67-ch67.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, covariance_type="full",
                     n_init=3, random_state=42).fit(X)

log_dens = gm.score_samples(X)
threshold = np.percentile(log_dens, 2)      # عتبة المئين الثاني
print(f"عتبة الكثافة (المئين 2): {threshold:.2f}")

outliers = np.array([[16.0, -14.0], [-17.0, 12.0], [14.0, 15.0],
                     [-15.0, -16.0], [18.0, 4.0]])
caught = (gm.score_samples(outliers) < threshold).sum()
print(f"نقاط شاذة محقونة: {len(outliers)} | الملتقطة تحت العتبة: {caught}")
print(f"نسبة الالتقاط: {caught / len(outliers):.0%}")
