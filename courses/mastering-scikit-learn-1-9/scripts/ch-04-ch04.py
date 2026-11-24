#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أول نموذج لك: أقرب الجيران على بيانات Iris
Mastering scikit-learn 1.9 — Iris

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-04-ch04 (lesson 4)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أول نموذج لك: أقرب الجيران على بيانات Iris
Mastering scikit-learn 1.9 — Iris

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-04-ch04 (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris, make_classification

iris = load_iris(as_frame=True)
df, X, y = iris.frame, iris.data, np.asarray(iris.target)

print(df.head(4).to_string())
print("\nالشكل:", df.shape, "  توزيع الأصناف:", np.bincount(y))
print("أسماء الأصناف:", list(iris.target_names))

print("\nمتوسّط كل صنف (الأرقام التي ستتعلّمها):")
means = df.groupby("target")[iris.feature_names].mean().round(2)
means.index = [n[:4].title() for n in iris.target_names]
print(means.to_string())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Xa, ya = np.asarray(X), y
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25,
                                              random_state=42, stratify=ya)
pl, pw = Xa_te[:, 2], Xa_te[:, 3]

rule_one = np.where(pl < 2.5, 0, 1)
rule_two = np.where(pl < 2.5, 0, np.where(pw < 1.75, 1, 2))

print("قاعدة واحدة (طول البتلات فقط) : %.4f" % (rule_one == ya_te).mean())
print("قاعدتان (طول وعرض البتلات)   : %.4f" % (rule_two == ya_te).mean())
print("مصفوفة الالتباس للقاعدتين     :")
print(confusion_matrix(ya_te, rule_two))
print("\nnearest neighbours, k=5        : %.4f"
      # % KNeighborsClassifier(n_neighbors=5).fit(Xa_tr, ya_tr).score(Xa_te, ya_te))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=42, stratify=y)
print("تدريب:", Xtr.shape, " اختبار:", Xte.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
y_pred = knn.predict(Xte)

print("\nالدقة:", round(accuracy_score(yte, y_pred), 4))
print("\nتقرير التصنيف:")
print(classification_report(yte, y_pred,
                            target_names=iris.target_names, digits=3))
print("مصفوفة الالتباس (صف = الحقيقي، عمود = المتوقّع):")
print(confusion_matrix(yte, y_pred))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
q = np.array([[5.0, 3.4, 1.5, 0.2]])          # قطرة جديدة
Xtr_np = np.asarray(Xtr)

distances = np.linalg.norm(Xtr_np - q, axis=1)
nearest = np.argsort(distances)[:3]

print("صفوف التدريب:", [int(i) for i in nearest])
print("المسافات     :", np.round(distances[nearest], 4))
print("أصنافها      :", ytr[nearest])

manual = np.bincount(ytr[nearest], minlength=3) / 3
library = KNeighborsClassifier(n_neighbors=3).fit(Xtr_np, ytr).predict_proba(q)[0]
print("\nاحتمالي يدوي  :", manual)
print("احتمالي المكتبة:", library)
print("مطابق تماماً    :", np.allclose(manual, library))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
print(f"{'k':>4} {'weights':>9} {'train acc':>10} {'test acc':>9}")
for k in (1, 3, 5, 9, 15, 25, 51):
    for w in ("uniform", "distance"):
        m = KNeighborsClassifier(n_neighbors=k, weights=w).fit(Xtr, ytr)
        print(f"{k:>4} {w:>9} {m.score(Xtr, ytr):>10.4f} {m.score(Xte, yte):>9.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xbig, ybig = make_classification(n_samples=1200, n_features=6, n_informative=2,
                                 n_redundant=0, n_repeated=0, random_state=5)
Xbig = np.asarray(Xbig)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

print(f"{'معامل عمود الضجيج':>20} {'خام':>9} {'مُحجَّم':>9}")
for mult in (0.01, 1.0, 100.0, 10_000.0):
    Xz = Xbig.copy()
    Xz[:, 5] = Xz[:, 5] * mult            # عمود ضجيج بوحدات مختلفة
    s_raw = cross_val_score(knn_raw, Xz, ybig, cv=cv, n_jobs=1).mean()
    s_scl = cross_val_score(knn_scaled, Xz, ybig, cv=cv, n_jobs=1).mean()
    print(f"{mult:>20g} {s_raw:>9.4f} {s_scl:>9.4f}")

print("\nوالمقارنة على Iris نفسها:")
Xa, ya = np.asarray(X), y
ir_tr, ir_te, iy_tr, iy_te = train_test_split(Xa, ya, test_size=0.25,
                                              random_state=42, stratify=ya)
print("مدى كل عمود            :", np.round(Xa.max(0) - Xa.min(0), 2))
print("k-NN خام على Iris      : %.4f"
      # % KNeighborsClassifier(n_neighbors=5).fit(ir_tr, iy_tr).score(ir_te, iy_te))
print("k-NN محجَّم على Iris   : %.4f" % knn_scaled.fit(ir_tr, iy_tr).score(ir_te, iy_te))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch04.md
# ────────────────────────────────────────────────────────────────────────
import time

for n in (500, 2_000, 8_000, 32_000):
    Xn, yn = make_classification(n_samples=n, n_features=6, n_informative=3,
                                 n_redundant=0, random_state=0)
    m = KNeighborsClassifier(n_neighbors=5).fit(Xn, yn)
    t0 = time.perf_counter()
    m.predict(Xn[:2_000])
    dt = time.perf_counter() - t0
    print(f"  n_train = {n:>6}   predict(2000) = {dt:.4f}s")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris, make_classification

iris = load_iris(as_frame=True)
df, X, y = iris.frame, iris.data, np.asarray(iris.target)

print(df.head(4).to_string())
print("\nالشكل:", df.shape, "  توزيع الأصناف:", np.bincount(y))
print("أسماء الأصناف:", list(iris.target_names))

print("\nمتوسّط كل صنف (الأرقام التي ستتعلّمها):")
means = df.groupby("target")[iris.feature_names].mean().round(2)
means.index = [n[:4].title() for n in iris.target_names]
print(means.to_string())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Xa, ya = np.asarray(X), y
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25,
                                              random_state=42, stratify=ya)
pl, pw = Xa_te[:, 2], Xa_te[:, 3]

rule_one = np.where(pl < 2.5, 0, 1)
rule_two = np.where(pl < 2.5, 0, np.where(pw < 1.75, 1, 2))

print("قاعدة واحدة (طول البتلات فقط) : %.4f" % (rule_one == ya_te).mean())
print("قاعدتان (طول وعرض البتلات)   : %.4f" % (rule_two == ya_te).mean())
print("مصفوفة الالتباس للقاعدتين     :")
print(confusion_matrix(ya_te, rule_two))
print("\nnearest neighbours, k=5        : %.4f"
      # % KNeighborsClassifier(n_neighbors=5).fit(Xa_tr, ya_tr).score(Xa_te, ya_te))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=42, stratify=y)
print("تدريب:", Xtr.shape, " اختبار:", Xte.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
y_pred = knn.predict(Xte)

print("\nالدقة:", round(accuracy_score(yte, y_pred), 4))
print("\nتقرير التصنيف:")
print(classification_report(yte, y_pred,
                            target_names=iris.target_names, digits=3))
print("مصفوفة الالتباس (صف = الحقيقي، عمود = المتوقّع):")
print(confusion_matrix(yte, y_pred))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
q = np.array([[5.0, 3.4, 1.5, 0.2]])          # قطرة جديدة
Xtr_np = np.asarray(Xtr)

distances = np.linalg.norm(Xtr_np - q, axis=1)
nearest = np.argsort(distances)[:3]

print("صفوف التدريب:", [int(i) for i in nearest])
print("المسافات     :", np.round(distances[nearest], 4))
print("أصنافها      :", ytr[nearest])

manual = np.bincount(ytr[nearest], minlength=3) / 3
library = KNeighborsClassifier(n_neighbors=3).fit(Xtr_np, ytr).predict_proba(q)[0]
print("\nاحتمالي يدوي  :", manual)
print("احتمالي المكتبة:", library)
print("مطابق تماماً    :", np.allclose(manual, library))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
print(f"{'k':>4} {'weights':>9} {'train acc':>10} {'test acc':>9}")
for k in (1, 3, 5, 9, 15, 25, 51):
    for w in ("uniform", "distance"):
        m = KNeighborsClassifier(n_neighbors=k, weights=w).fit(Xtr, ytr)
        print(f"{k:>4} {w:>9} {m.score(Xtr, ytr):>10.4f} {m.score(Xte, yte):>9.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xbig, ybig = make_classification(n_samples=1200, n_features=6, n_informative=2,
                                 n_redundant=0, n_repeated=0, random_state=5)
Xbig = np.asarray(Xbig)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

print(f"{'معامل عمود الضجيج':>20} {'خام':>9} {'مُحجَّم':>9}")
for mult in (0.01, 1.0, 100.0, 10_000.0):
    Xz = Xbig.copy()
    Xz[:, 5] = Xz[:, 5] * mult            # عمود ضجيج بوحدات مختلفة
    s_raw = cross_val_score(knn_raw, Xz, ybig, cv=cv, n_jobs=1).mean()
    s_scl = cross_val_score(knn_scaled, Xz, ybig, cv=cv, n_jobs=1).mean()
    print(f"{mult:>20g} {s_raw:>9.4f} {s_scl:>9.4f}")

print("\nوالمقارنة على Iris نفسها:")
Xa, ya = np.asarray(X), y
ir_tr, ir_te, iy_tr, iy_te = train_test_split(Xa, ya, test_size=0.25,
                                              random_state=42, stratify=ya)
print("مدى كل عمود            :", np.round(Xa.max(0) - Xa.min(0), 2))
print("k-NN خام على Iris      : %.4f"
      # % KNeighborsClassifier(n_neighbors=5).fit(ir_tr, iy_tr).score(ir_te, iy_te))
print("k-NN محجَّم على Iris   : %.4f" % knn_scaled.fit(ir_tr, iy_tr).score(ir_te, iy_te))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-04-ch04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import time

for n in (500, 2_000, 8_000, 32_000):
    Xn, yn = make_classification(n_samples=n, n_features=6, n_informative=3,
                                 n_redundant=0, random_state=0)
    m = KNeighborsClassifier(n_neighbors=5).fit(Xn, yn)
    t0 = time.perf_counter()
    m.predict(Xn[:2_000])
    dt = time.perf_counter() - t0
    print(f"  n_train = {n:>6}   predict(2000) = {dt:.4f}s")
