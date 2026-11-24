#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — القيم المفقودة: من الفراغ إلى المعلومة
Mastering scikit-learn 1.9 — Chapter 13 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-13-ch13 (lesson 13)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-13-ch13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — القيم المفقودة: من الفراغ إلى المعلومة
Mastering scikit-learn 1.9 — Chapter 13 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-13-ch13 (lesson 13)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

rng = np.random.default_rng(0)
holes = rng.random(df.shape) < 0.08
df[holes] = np.nan

print("عدد الفجوات في كل عمود:")
print(df.isna().sum().to_string())
print()
print(f"نسبة الفجوات الكلية     : {df.isna().to_numpy().mean():.3f}")
print(f"صفوف فيها فجوة واحدة على الأقل: {int(df.isna().any(axis=1).sum())} من {len(df)}")
print(f"صفوف فيها فجوتان أو أكثر: {int((df.isna().sum(axis=1) >= 2).sum())}")
print(f"صفوف كاملة الفقد        : {int((df.isna().sum(axis=1) == df.shape[1]).sum())}")

print("\n--- الآن اصنع فجوة من نوع MAR: احفرها داخل الصنف الثالث فقط ---")
mar = iris.data.copy()
mar[np.where(iris.target == 2)[0], 2] = np.nan
print(f"نسبة الفجوة في العمود الثالث: {np.isnan(mar).mean():.3f}")

imputed = SimpleImputer(strategy="mean").fit_transform(mar).to_numpy()
print("عدد NaN بعد التعبئة بالمتوسّط:", int(np.isnan(imputed).sum()))

before = [iris.data[iris.target == k, 2].mean() for k in range(3)]
after = [imputed[iris.target == k, 2].mean() for k in range(3)]
print("\nمتوسط طول البتلات لكل صنف:")
print("  قبل التعبئة:", np.round(before, 3).tolist())
print("  بعد التعبئة:", np.round(after, 3).tolist())
print("  الفارق      :", np.round(np.array(after) - np.array(before), 3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
Xm = np.array([[1.0, 2.0],
               [np.nan, 3.0],
               [4.0, np.nan],
               [5.0, 6.0],
               [np.nan, np.nan]])

print("المصفوفة الأصلية:")
print(Xm)
for st in ["mean", "median", "most_frequent"]:
    s = SimpleImputer(strategy=st).fit(Xm)
    print(f"\n{st}: القيم البديلة = {np.round(s.statistics_, 3).tolist()}")
    print(s.transform(Xm).to_numpy())

s_const = SimpleImputer(strategy="constant", fill_value=-999).fit(Xm)
print("\nconstant=-999:")
print(s_const.transform(Xm).to_numpy())

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
import sklearn.experimental.enable_iterative_imputer  # noqa: F401 Side effect only
from sklearn.impute import IterativeImputer          # الآن ينجح

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.impute import KNNImputer
import sklearn.experimental.enable_iterative_imputer  # noqa: F401

IterativeImputer = sklearn.impute.IterativeImputer

print("KNNImputer (الجيران):")
print(np.round(KNNImputer(n_neighbors=2).fit_transform(Xm).to_numpy(), 3))

print("\nIterativeImputer (نموذج يتكرّر):")
it = IterativeImputer(max_iter=25, tol=1e-3, random_state=0)
print(np.round(it.fit_transform(Xm).to_numpy(), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.impute import MissingIndicator

ind_all = MissingIndicator(features="all").fit_transform(Xm)
ind_missing = MissingIndicator(features="missing-only").fit_transform(Xm)
print("features='all'          :")
print(ind_all.to_numpy().astype(int))
print("features='missing-only'  :")
print(ind_missing.to_numpy().astype(int))
print("نوع القيم:", ind_all.dtypes.tolist())

filled = SimpleImputer(strategy="mean", add_indicator=True).fit_transform(Xm)
print("\nمع add_indicator=True، أعمدة التعبئة ثم الأعمدة الثنائية:")
print(filled.to_numpy().round(3))

Xall = np.array([[1.0, np.nan], [2.0, np.nan]])
print("\nالفخّ الصامت: عمود كامل الفراغ")
print("  keep_empty_features=False:", SimpleImputer().fit_transform(Xall).shape)
print("  keep_empty_features=True :",
      SimpleImputer(keep_empty_features=True).fit_transform(Xall).shape)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
big = np.random.default_rng(3).normal(10, 2, 2000)
big[:400] = np.nan           # ٢٠٪ فراغ

filled_mean = SimpleImputer(strategy="mean").fit_transform(big.reshape(-1, 1)).to_numpy().ravel()
filled_median = SimpleImputer(strategy="median").fit_transform(big.reshape(-1, 1)).to_numpy().ravel()

print(f"قبل التعبئة : mean={np.nanmean(big):7.3f}  std={np.nanstd(big):6.3f}")
print(f"بعد mean     : mean={filled_mean.mean():7.3f}  std={filled_mean.std():6.3f}")
print(f"بعد median   : mean={filled_median.mean():7.3f}  std={filled_median.std():6.3f}")
loss = 100 * (1 - filled_mean.std() / np.nanstd(big))
print(f"\nالانكماش المقاس: {loss:.1f}%  (النظرية: 20% لأن f = 0.20)")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch13.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

digits = load_digits()
Xd, yd = digits.data.copy(), digits.target
hole = np.random.default_rng(1).random(Xd.shape) < 0.10
Xd[hole] = np.nan

Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
print(f"عيّنات التدريب {Xtr.shape}، الاختبار {Xte.shape}")
print(f"نسبة الفجوات في التدريب: {np.isnan(Xtr).mean():.3f}، في الاختبار: {np.isnan(Xte).mean():.3f}")

try:
    KNeighborsClassifier(5).fit(Xtr, ytr)
except Exception as exc:
    print("بلا معالجة، النموذج يقول:", type(exc).__name__, "-", str(exc).splitlines()[0][:60])


def measure(name, step):
    pipe = Pipeline([("imp", step), ("knn", KNeighborsClassifier(5))])
    pipe.fit(Xtr, ytr)
    return pipe.score(Xte, yte)


for name, step in [
    ("mean", SimpleImputer(strategy="mean")),
    ("median", SimpleImputer(strategy="median")),
    ("constant=-1", SimpleImputer(strategy="constant", fill_value=-1)),
    ("most_frequent", SimpleImputer(strategy="most_frequent")),
    ("KNNImputer(5)", KNNImputer(n_neighbors=5)),
    ("IterativeImputer", IterativeImputer(max_iter=10, random_state=0)),
    ("mean+indicator", SimpleImputer(strategy="mean", add_indicator=True)),
]:
    print(f"  {name:18s} test accuracy = {measure(name, step):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

rng = np.random.default_rng(0)
holes = rng.random(df.shape) < 0.08
df[holes] = np.nan

print("عدد الفجوات في كل عمود:")
print(df.isna().sum().to_string())
print()
print(f"نسبة الفجوات الكلية     : {df.isna().to_numpy().mean():.3f}")
print(f"صفوف فيها فجوة واحدة على الأقل: {int(df.isna().any(axis=1).sum())} من {len(df)}")
print(f"صفوف فيها فجوتان أو أكثر: {int((df.isna().sum(axis=1) >= 2).sum())}")
print(f"صفوف كاملة الفقد        : {int((df.isna().sum(axis=1) == df.shape[1]).sum())}")

print("\n--- الآن اصنع فجوة من نوع MAR: احفرها داخل الصنف الثالث فقط ---")
mar = iris.data.copy()
mar[np.where(iris.target == 2)[0], 2] = np.nan
print(f"نسبة الفجوة في العمود الثالث: {np.isnan(mar).mean():.3f}")

imputed = SimpleImputer(strategy="mean").fit_transform(mar).to_numpy()
print("عدد NaN بعد التعبئة بالمتوسّط:", int(np.isnan(imputed).sum()))

before = [iris.data[iris.target == k, 2].mean() for k in range(3)]
after = [imputed[iris.target == k, 2].mean() for k in range(3)]
print("\nمتوسط طول البتلات لكل صنف:")
print("  قبل التعبئة:", np.round(before, 3).tolist())
print("  بعد التعبئة:", np.round(after, 3).tolist())
print("  الفارق      :", np.round(np.array(after) - np.array(before), 3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
Xm = np.array([[1.0, 2.0],
               [np.nan, 3.0],
               [4.0, np.nan],
               [5.0, 6.0],
               [np.nan, np.nan]])

print("المصفوفة الأصلية:")
print(Xm)
for st in ["mean", "median", "most_frequent"]:
    s = SimpleImputer(strategy=st).fit(Xm)
    print(f"\n{st}: القيم البديلة = {np.round(s.statistics_, 3).tolist()}")
    print(s.transform(Xm).to_numpy())

s_const = SimpleImputer(strategy="constant", fill_value=-999).fit(Xm)
print("\nconstant=-999:")
print(s_const.transform(Xm).to_numpy())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import sklearn.experimental.enable_iterative_imputer  # noqa: F401 Side effect only
from sklearn.impute import IterativeImputer          # الآن ينجح

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.impute import KNNImputer
import sklearn.experimental.enable_iterative_imputer  # noqa: F401

IterativeImputer = sklearn.impute.IterativeImputer

print("KNNImputer (الجيران):")
print(np.round(KNNImputer(n_neighbors=2).fit_transform(Xm).to_numpy(), 3))

print("\nIterativeImputer (نموذج يتكرّر):")
it = IterativeImputer(max_iter=25, tol=1e-3, random_state=0)
print(np.round(it.fit_transform(Xm).to_numpy(), 3))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.impute import MissingIndicator

ind_all = MissingIndicator(features="all").fit_transform(Xm)
ind_missing = MissingIndicator(features="missing-only").fit_transform(Xm)
print("features='all'          :")
print(ind_all.to_numpy().astype(int))
print("features='missing-only'  :")
print(ind_missing.to_numpy().astype(int))
print("نوع القيم:", ind_all.dtypes.tolist())

filled = SimpleImputer(strategy="mean", add_indicator=True).fit_transform(Xm)
print("\nمع add_indicator=True، أعمدة التعبئة ثم الأعمدة الثنائية:")
print(filled.to_numpy().round(3))

Xall = np.array([[1.0, np.nan], [2.0, np.nan]])
print("\nالفخّ الصامت: عمود كامل الفراغ")
print("  keep_empty_features=False:", SimpleImputer().fit_transform(Xall).shape)
print("  keep_empty_features=True :",
      SimpleImputer(keep_empty_features=True).fit_transform(Xall).shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
big = np.random.default_rng(3).normal(10, 2, 2000)
big[:400] = np.nan           # ٢٠٪ فراغ

filled_mean = SimpleImputer(strategy="mean").fit_transform(big.reshape(-1, 1)).to_numpy().ravel()
filled_median = SimpleImputer(strategy="median").fit_transform(big.reshape(-1, 1)).to_numpy().ravel()

print(f"قبل التعبئة : mean={np.nanmean(big):7.3f}  std={np.nanstd(big):6.3f}")
print(f"بعد mean     : mean={filled_mean.mean():7.3f}  std={filled_mean.std():6.3f}")
print(f"بعد median   : mean={filled_median.mean():7.3f}  std={filled_median.std():6.3f}")
loss = 100 * (1 - filled_mean.std() / np.nanstd(big))
print(f"\nالانكماش المقاس: {loss:.1f}%  (النظرية: 20% لأن f = 0.20)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-13-ch13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

digits = load_digits()
Xd, yd = digits.data.copy(), digits.target
hole = np.random.default_rng(1).random(Xd.shape) < 0.10
Xd[hole] = np.nan

Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
print(f"عيّنات التدريب {Xtr.shape}، الاختبار {Xte.shape}")
print(f"نسبة الفجوات في التدريب: {np.isnan(Xtr).mean():.3f}، في الاختبار: {np.isnan(Xte).mean():.3f}")

try:
    KNeighborsClassifier(5).fit(Xtr, ytr)
except Exception as exc:
    print("بلا معالجة، النموذج يقول:", type(exc).__name__, "-", str(exc).splitlines()[0][:60])


def measure(name, step):
    pipe = Pipeline([("imp", step), ("knn", KNeighborsClassifier(5))])
    pipe.fit(Xtr, ytr)
    return pipe.score(Xte, yte)


for name, step in [
    ("mean", SimpleImputer(strategy="mean")),
    ("median", SimpleImputer(strategy="median")),
    ("constant=-1", SimpleImputer(strategy="constant", fill_value=-1)),
    ("most_frequent", SimpleImputer(strategy="most_frequent")),
    ("KNNImputer(5)", KNNImputer(n_neighbors=5)),
    ("IterativeImputer", IterativeImputer(max_iter=10, random_state=0)),
    ("mean+indicator", SimpleImputer(strategy="mean", add_indicator=True)),
]:
    print(f"  {name:18s} test accuracy = {measure(name, step):.4f}")
