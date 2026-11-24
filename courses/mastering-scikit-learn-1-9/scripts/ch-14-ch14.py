#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحجيم والتطبيع
Mastering scikit-learn 1.9 — Chapter 14 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-14-ch14 (lesson 14)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-14-ch14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحجيم والتطبيع
Mastering scikit-learn 1.9 — Chapter 14 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-14-ch14 (lesson 14)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "salary": rng.normal(12000, 2500, 200),
    "age": rng.normal(35, 6, 200),
})

sc = StandardScaler().fit(df)
print("المتوسّطات المحفوظة mean_ :", np.round(sc.mean_, 2))
print("الانحرافات المحفوظة scale_:", np.round(sc.scale_, 2))

Z = sc.transform(df)
print("بعد التحويل -> mean:", np.round(Z.mean().to_numpy(), 6))
print("بعد التحويل -> std :", np.round(Z.std().to_numpy(), 6))
print("أقصى خطأ في inverse_transform:", float(np.abs(sc.inverse_transform(Z) - df.to_numpy()).max()))
print("تكرار transform بنفس mean_:", float(np.abs(sc.transform(df) - Z).to_numpy().max()))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

mm = MinMaxScaler().fit(df)
print("MinMaxScaler:")
print("  range_ الافتراضي :", mm.data_range_.round(2))
print("  ناتج التحويل      : من", mm.transform(df).min().to_numpy().round(3),
      "إلى", mm.transform(df).max().to_numpy().round(3))

mm11 = MinMaxScaler(feature_range=(-1, 1)).fit(df)
print("  feature_range=(-1,1): من", mm11.transform(df).min().to_numpy().round(3),
      "إلى", mm11.transform(df).max().to_numpy().round(3))

ma = MaxAbsScaler().fit(df)
print("\nMaxAbsScaler scale_:", np.round(ma.scale_, 2))
print("  ناتج التحويل      : من", ma.transform(df).min().to_numpy().round(3),
      "إلى", ma.transform(df).max().to_numpy().round(3))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler

polluted = df.copy()
polluted.loc[:4, "salary"] = 90000

rs = RobustScaler().fit(polluted)
ss = StandardScaler().fit(polluted)
print("بدون شواذ  -> Robust center:", round(RobustScaler().fit(df).center_[0], 1))
print("مع شواذ    -> Robust center:", round(rs.center_[0], 1))
print("مع شواذ    -> Standard mean :", round(ss.mean_[0], 1))
print()
print("Robust  scale_ :", np.round(rs.scale_, 2), " (المدى الربيعي)")
print("Standard scale_:", np.round(ss.scale_, 2))
print("أكبر قيمة Robust   :", round(float(rs.transform(polluted).to_numpy()[:, 0].max()), 2))
print("أكبر قيمة Standard:", round(float(ss.transform(polluted).to_numpy()[:, 0].max()), 2))
print()
print("مع quantile_range=(10, 90) :", np.round(RobustScaler(quantile_range=(10.0, 90.0)).fit(polluted).scale_, 2))
print("مع unit_variance=True      :", np.round(RobustScaler(unit_variance=True).fit(polluted).scale_, 3))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import Normalizer

A = np.array([[3.0, 4.0],
              [1.0, 0.0],
              [0.0, 0.0]])
print("قبل التطبيع:", A.tolist())
for norm in ["l1", "l2", "max"]:
    out = Normalizer(norm=norm).fit_transform(A).to_numpy()
    print(f"  norm={norm:4s} -> {np.round(out, 3).tolist()}   (طول الصفّ = {np.round(np.abs(out).sum(axis=1), 3).tolist()})")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import KernelCenterer

K = np.array([[1.0, 0.5],
              [0.5, 1.0]])
kc = KernelCenterer().fit(K)
print("قبل:\n", K)
print("بعد التوسيط:\n", kc.transform(K).round(4))
print("المصفوفة المحفوظة K_fit_all_:\n", np.round(kc.K_fit_all_, 4))
try:
    KernelCenterer().fit_transform(df)
except Exception as exc:
    print("\nعلى جدول 200x2 ->", type(exc).__name__, ":", str(exc).splitlines()[0][:70])

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

Xb, yb = make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=0)
Xb = Xb * np.array([1.0, 120.0])          # ميزانان بمقياسين مختلفين تماماً
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=0, stratify=yb)

print("متوسّطات التدريب:", np.round(Xtr.mean(axis=0), 2), " انحرافاتها:", np.round(Xtr.std(axis=0), 2))
S = StandardScaler().fit_transform(Xtr).to_numpy()
d0 = np.linalg.norm(Xtr[0] - Xtr[1])
d1 = np.linalg.norm(S[0] - S[1])
print("مسافة بين عيّنتين: خام %.2f | بعد التحجيم %.3f" % (d0, d1))
print("  مساهمة كلّ محور خام:", np.abs(Xtr[0] - Xtr[1]).round(2))
print("  مساهمة كلّ محور بعد التحجيم:", np.abs(S[0] - S[1]).round(3))
print()

knn = KNeighborsClassifier(5)
print("kNN خام    :", round(knn.fit(Xtr, ytr).score(Xte, yte), 4))
scaled_knn = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))]).fit(Xtr, ytr)
print("kNN محجَّم :", round(scaled_knn.score(Xte, yte), 4))
tree = DecisionTreeClassifier(random_state=0)
print("شجرة خام   :", round(tree.fit(Xtr, ytr).score(Xte, yte), 4))
scaled_tree = Pipeline([("sc", StandardScaler()),
                        ("t", DecisionTreeClassifier(random_state=0))]).fit(Xtr, ytr)
print("شجرة محجَّمة:", round(scaled_tree.score(Xte, yte), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch14.md
# ────────────────────────────────────────────────────────────────────────
scaler.fit(X_all)            # خطأ: يستخدم كل البيانات
Z_all = scaler.transform(X_all)   # المتوسّط صار يعرف الاختبار
Xtr, Xte = split(Z_all)            # تسرّب من الأرقام لا من الصفوف

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "salary": rng.normal(12000, 2500, 200),
    "age": rng.normal(35, 6, 200),
})

sc = StandardScaler().fit(df)
print("المتوسّطات المحفوظة mean_ :", np.round(sc.mean_, 2))
print("الانحرافات المحفوظة scale_:", np.round(sc.scale_, 2))

Z = sc.transform(df)
print("بعد التحويل -> mean:", np.round(Z.mean().to_numpy(), 6))
print("بعد التحويل -> std :", np.round(Z.std().to_numpy(), 6))
print("أقصى خطأ في inverse_transform:", float(np.abs(sc.inverse_transform(Z) - df.to_numpy()).max()))
print("تكرار transform بنفس mean_:", float(np.abs(sc.transform(df) - Z).to_numpy().max()))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

mm = MinMaxScaler().fit(df)
print("MinMaxScaler:")
print("  range_ الافتراضي :", mm.data_range_.round(2))
print("  ناتج التحويل      : من", mm.transform(df).min().to_numpy().round(3),
      "إلى", mm.transform(df).max().to_numpy().round(3))

mm11 = MinMaxScaler(feature_range=(-1, 1)).fit(df)
print("  feature_range=(-1,1): من", mm11.transform(df).min().to_numpy().round(3),
      "إلى", mm11.transform(df).max().to_numpy().round(3))

ma = MaxAbsScaler().fit(df)
print("\nMaxAbsScaler scale_:", np.round(ma.scale_, 2))
print("  ناتج التحويل      : من", ma.transform(df).min().to_numpy().round(3),
      "إلى", ma.transform(df).max().to_numpy().round(3))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler

polluted = df.copy()
polluted.loc[:4, "salary"] = 90000

rs = RobustScaler().fit(polluted)
ss = StandardScaler().fit(polluted)
print("بدون شواذ  -> Robust center:", round(RobustScaler().fit(df).center_[0], 1))
print("مع شواذ    -> Robust center:", round(rs.center_[0], 1))
print("مع شواذ    -> Standard mean :", round(ss.mean_[0], 1))
print()
print("Robust  scale_ :", np.round(rs.scale_, 2), " (المدى الربيعي)")
print("Standard scale_:", np.round(ss.scale_, 2))
print("أكبر قيمة Robust   :", round(float(rs.transform(polluted).to_numpy()[:, 0].max()), 2))
print("أكبر قيمة Standard:", round(float(ss.transform(polluted).to_numpy()[:, 0].max()), 2))
print()
print("مع quantile_range=(10, 90) :", np.round(RobustScaler(quantile_range=(10.0, 90.0)).fit(polluted).scale_, 2))
print("مع unit_variance=True      :", np.round(RobustScaler(unit_variance=True).fit(polluted).scale_, 3))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import Normalizer

A = np.array([[3.0, 4.0],
              [1.0, 0.0],
              [0.0, 0.0]])
print("قبل التطبيع:", A.tolist())
for norm in ["l1", "l2", "max"]:
    out = Normalizer(norm=norm).fit_transform(A).to_numpy()
    print(f"  norm={norm:4s} -> {np.round(out, 3).tolist()}   (طول الصفّ = {np.round(np.abs(out).sum(axis=1), 3).tolist()})")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import KernelCenterer

K = np.array([[1.0, 0.5],
              [0.5, 1.0]])
kc = KernelCenterer().fit(K)
print("قبل:\n", K)
print("بعد التوسيط:\n", kc.transform(K).round(4))
print("المصفوفة المحفوظة K_fit_all_:\n", np.round(kc.K_fit_all_, 4))
try:
    KernelCenterer().fit_transform(df)
except Exception as exc:
    print("\nعلى جدول 200x2 ->", type(exc).__name__, ":", str(exc).splitlines()[0][:70])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

Xb, yb = make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=0)
Xb = Xb * np.array([1.0, 120.0])          # ميزانان بمقياسين مختلفين تماماً
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=0, stratify=yb)

print("متوسّطات التدريب:", np.round(Xtr.mean(axis=0), 2), " انحرافاتها:", np.round(Xtr.std(axis=0), 2))
S = StandardScaler().fit_transform(Xtr).to_numpy()
d0 = np.linalg.norm(Xtr[0] - Xtr[1])
d1 = np.linalg.norm(S[0] - S[1])
print("مسافة بين عيّنتين: خام %.2f | بعد التحجيم %.3f" % (d0, d1))
print("  مساهمة كلّ محور خام:", np.abs(Xtr[0] - Xtr[1]).round(2))
print("  مساهمة كلّ محور بعد التحجيم:", np.abs(S[0] - S[1]).round(3))
print()

knn = KNeighborsClassifier(5)
print("kNN خام    :", round(knn.fit(Xtr, ytr).score(Xte, yte), 4))
scaled_knn = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))]).fit(Xtr, ytr)
print("kNN محجَّم :", round(scaled_knn.score(Xte, yte), 4))
tree = DecisionTreeClassifier(random_state=0)
print("شجرة خام   :", round(tree.fit(Xtr, ytr).score(Xte, yte), 4))
scaled_tree = Pipeline([("sc", StandardScaler()),
                        ("t", DecisionTreeClassifier(random_state=0))]).fit(Xtr, ytr)
print("شجرة محجَّمة:", round(scaled_tree.score(Xte, yte), 4))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-14-ch14.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
scaler.fit(X_all)            # خطأ: يستخدم كل البيانات
Z_all = scaler.transform(X_all)   # المتوسّط صار يعرف الاختبار
Xtr, Xte = split(Z_all)            # تسرّب من الأرقام لا من الصفوف
