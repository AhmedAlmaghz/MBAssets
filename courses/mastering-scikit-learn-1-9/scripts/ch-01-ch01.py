#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ما هو تعلّم الآلة؟ رحلة من الفكرة إلى النظام
Mastering scikit-learn 1.9 — Chapter 01 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-01-ch01 (lesson 1)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-01-ch01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ما هو تعلّم الآلة؟ رحلة من الفكرة إلى النظام
Mastering scikit-learn 1.9 — Chapter 01 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-01-ch01 (lesson 1)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Xc, yc = make_classification(n_samples=1200, n_features=6, n_informative=4,
                             n_redundant=1, class_sep=1.4, random_state=42)
Xr, yr = make_regression(n_samples=800, n_features=5, noise=12.0,
                         random_state=42)

print("classification :", Xc.shape, "-> class counts", np.bincount(yc))
print("regression     :", Xr.shape, "-> target mean %.2f  std %.2f"
      # % (yr.mean(), yr.std()))
print("feature means  :", np.round(Xc.mean(axis=0), 3))
print("feature stds   :", np.round(Xc.std(axis=0), 3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(3)
ang = rng.uniform(0, 2 * np.pi, 600)
rad = rng.uniform(0.3, 3.0, 600)
Xc = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
yc = (rad > 2.0).astype(int)

rule = (np.hypot(Xc[:, 0], Xc[:, 1]) > 2.0).astype(int)
logreg = LogisticRegression(max_iter=2000).fit(Xc, yc)
print("حدّ قرار دائري مثالي:")
print("  قاعدة بالمسافة : %.4f" % (rule == yc).mean())
print("  خط مستقيم       : %.4f" % logreg.score(Xc, yc))

xq, yq = make_regression(n_samples=60, n_features=1, noise=6.0, random_state=0)
xq = xq.ravel()
med = np.median(xq)
median_pred = np.where(xq <= med, yq[xq <= med].mean(), yq[xq > med].mean())
lin = LinearRegression().fit(xq.reshape(-1, 1), yq)
print("\nمسألة بعلاقة نعّاحة وضجيج:")
print("  قاعدة الوسيط    : R2 = %.4f" % r2_score(yq, median_pred))
print("  الانحدار الخطي  : R2 = %.4f" % r2_score(yq, lin.predict(xq.reshape(-1, 1))))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

Xu, truth = make_blobs(n_samples=420, centers=4, cluster_std=0.7, random_state=42)
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xu)
db = DBSCAN(eps=0.9, min_samples=8).fit(Xu)
n_db = len(set(db.labels_)) - int(-1 in db.labels_)

print(f"عدد المجموعات المكتشفة : KMeans {len(np.unique(km.labels_))}"
      f"   DBSCAN {n_db}   (الحقيقة {len(np.unique(truth))})")
print(f"adjusted_rand_score    : KMeans {adjusted_rand_score(truth, km.labels_):.4f}"
      f"   DBSCAN {adjusted_rand_score(truth, db.labels_):.4f}")
print(f"silhouette_score       : KMeans {silhouette_score(Xu, km.labels_):.4f}"
      f"   DBSCAN {silhouette_score(Xu, db.labels_):.4f}")
print(f"نقاط وصفها DBSCAN كشذوذ: {int((db.labels_ == -1).sum())}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "DummyClassifier(strategy='most_frequent')": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression(max_iter=2000)": LogisticRegression(max_iter=2000),
    "KNeighborsClassifier(n_neighbors=5)": KNeighborsClassifier(n_neighbors=5),
}
print(f"{'model':<38} {'cv accuracy':>12} {'std':>8}")
for name, est in candidates.items():
    s = cross_val_score(est, Xc, yc, cv=cv, scoring="accuracy")
    print(f"{name:<38} {s.mean():>12.4f} {s.std():>8.4f}")

lr_model = candidates["LogisticRegression(max_iter=2000)"]
sizes, train_scores, val_scores = learning_curve(
    lr_model, Xc, yc, cv=cv, train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy", n_jobs=1)

print("\nمنحنى التعلّم للانحدار اللوجيستي:")
print(pd.DataFrame({
    "n_train": sizes.astype(int),
    "train_acc": train_scores.mean(axis=1).round(4),
    "val_acc": val_scores.mean(axis=1).round(4),
    "gap": (train_scores.mean(axis=1) - val_scores.mean(axis=1)).round(4),
}).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Xr, yr = make_regression(n_samples=120, n_features=3, noise=28.0, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=42)
print(f"train {Xtr.shape}   test {Xte.shape}   target std {yr.std():.2f}")

print(f"\n{'degree':>7} {'n_params':>10} {'train R2':>10} {'test R2':>10}")
for deg in (1, 3, 12):
    pipe = make_pipeline(PolynomialFeatures(degree=deg),
                         StandardScaler(),
                         Ridge(alpha=1.0))
    pipe.fit(Xtr, ytr)
    n_params = len(pipe[-1].coef_) + 1
    print(f"{deg:>7} {n_params:>10} {pipe.score(Xtr, ytr):>10.4f} "
          f"{pipe.score(Xte, yte):>10.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch01.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

Xm, ym = make_moons(n_samples=300, noise=0.45, random_state=42)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3,
                                               random_state=42, stratify=ym)
knn_pipe = make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=7))
knn_pipe.fit(Xm_tr, ym_tr)

print("خط الأنابيب:", " -> ".join(name for name, _ in knn_pipe.steps))
print("تدريب %.4f   اختبار %.4f" % (knn_pipe.score(Xm_tr, ym_tr),
                                     knn_pipe.score(Xm_te, ym_te)))
print("احتمال الصنف 1 لأول ثلاث نقاط:",
      np.round(knn_pipe.predict_proba(Xm_te[:3])[:, 1], 3))

print("\nأثر اختيار الجيران:")
for k in (1, 5, 25):
    knn_pipe[-1].set_params(n_neighbors=k)
    print(f"  k={k:>2}  تدريب {knn_pipe.score(Xm_tr, ym_tr):.4f}"
          f"   اختبار {knn_pipe.score(Xm_te, ym_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Xc, yc = make_classification(n_samples=1200, n_features=6, n_informative=4,
                             n_redundant=1, class_sep=1.4, random_state=42)
Xr, yr = make_regression(n_samples=800, n_features=5, noise=12.0,
                         random_state=42)

print("classification :", Xc.shape, "-> class counts", np.bincount(yc))
print("regression     :", Xr.shape, "-> target mean %.2f  std %.2f"
      # % (yr.mean(), yr.std()))
print("feature means  :", np.round(Xc.mean(axis=0), 3))
print("feature stds   :", np.round(Xc.std(axis=0), 3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(3)
ang = rng.uniform(0, 2 * np.pi, 600)
rad = rng.uniform(0.3, 3.0, 600)
Xc = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
yc = (rad > 2.0).astype(int)

rule = (np.hypot(Xc[:, 0], Xc[:, 1]) > 2.0).astype(int)
logreg = LogisticRegression(max_iter=2000).fit(Xc, yc)
print("حدّ قرار دائري مثالي:")
print("  قاعدة بالمسافة : %.4f" % (rule == yc).mean())
print("  خط مستقيم       : %.4f" % logreg.score(Xc, yc))

xq, yq = make_regression(n_samples=60, n_features=1, noise=6.0, random_state=0)
xq = xq.ravel()
med = np.median(xq)
median_pred = np.where(xq <= med, yq[xq <= med].mean(), yq[xq > med].mean())
lin = LinearRegression().fit(xq.reshape(-1, 1), yq)
print("\nمسألة بعلاقة نعّاحة وضجيج:")
print("  قاعدة الوسيط    : R2 = %.4f" % r2_score(yq, median_pred))
print("  الانحدار الخطي  : R2 = %.4f" % r2_score(yq, lin.predict(xq.reshape(-1, 1))))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

Xu, truth = make_blobs(n_samples=420, centers=4, cluster_std=0.7, random_state=42)
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xu)
db = DBSCAN(eps=0.9, min_samples=8).fit(Xu)
n_db = len(set(db.labels_)) - int(-1 in db.labels_)

print(f"عدد المجموعات المكتشفة : KMeans {len(np.unique(km.labels_))}"
      f"   DBSCAN {n_db}   (الحقيقة {len(np.unique(truth))})")
print(f"adjusted_rand_score    : KMeans {adjusted_rand_score(truth, km.labels_):.4f}"
      f"   DBSCAN {adjusted_rand_score(truth, db.labels_):.4f}")
print(f"silhouette_score       : KMeans {silhouette_score(Xu, km.labels_):.4f}"
      f"   DBSCAN {silhouette_score(Xu, db.labels_):.4f}")
print(f"نقاط وصفها DBSCAN كشذوذ: {int((db.labels_ == -1).sum())}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "DummyClassifier(strategy='most_frequent')": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression(max_iter=2000)": LogisticRegression(max_iter=2000),
    "KNeighborsClassifier(n_neighbors=5)": KNeighborsClassifier(n_neighbors=5),
}
print(f"{'model':<38} {'cv accuracy':>12} {'std':>8}")
for name, est in candidates.items():
    s = cross_val_score(est, Xc, yc, cv=cv, scoring="accuracy")
    print(f"{name:<38} {s.mean():>12.4f} {s.std():>8.4f}")

lr_model = candidates["LogisticRegression(max_iter=2000)"]
sizes, train_scores, val_scores = learning_curve(
    lr_model, Xc, yc, cv=cv, train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy", n_jobs=1)

print("\nمنحنى التعلّم للانحدار اللوجيستي:")
print(pd.DataFrame({
    "n_train": sizes.astype(int),
    "train_acc": train_scores.mean(axis=1).round(4),
    "val_acc": val_scores.mean(axis=1).round(4),
    "gap": (train_scores.mean(axis=1) - val_scores.mean(axis=1)).round(4),
}).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Xr, yr = make_regression(n_samples=120, n_features=3, noise=28.0, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=42)
print(f"train {Xtr.shape}   test {Xte.shape}   target std {yr.std():.2f}")

print(f"\n{'degree':>7} {'n_params':>10} {'train R2':>10} {'test R2':>10}")
for deg in (1, 3, 12):
    pipe = make_pipeline(PolynomialFeatures(degree=deg),
                         StandardScaler(),
                         Ridge(alpha=1.0))
    pipe.fit(Xtr, ytr)
    n_params = len(pipe[-1].coef_) + 1
    print(f"{deg:>7} {n_params:>10} {pipe.score(Xtr, ytr):>10.4f} "
          f"{pipe.score(Xte, yte):>10.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-01-ch01.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

Xm, ym = make_moons(n_samples=300, noise=0.45, random_state=42)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3,
                                               random_state=42, stratify=ym)
knn_pipe = make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=7))
knn_pipe.fit(Xm_tr, ym_tr)

print("خط الأنابيب:", " -> ".join(name for name, _ in knn_pipe.steps))
print("تدريب %.4f   اختبار %.4f" % (knn_pipe.score(Xm_tr, ym_tr),
                                     knn_pipe.score(Xm_te, ym_te)))
print("احتمال الصنف 1 لأول ثلاث نقاط:",
      np.round(knn_pipe.predict_proba(Xm_te[:3])[:, 1], 3))

print("\nأثر اختيار الجيران:")
for k in (1, 5, 25):
    knn_pipe[-1].set_params(n_neighbors=k)
    print(f"  k={k:>2}  تدريب {knn_pipe.score(Xm_tr, ym_tr):.4f}"
          f"   اختبار {knn_pipe.score(Xm_te, ym_te):.4f}")
