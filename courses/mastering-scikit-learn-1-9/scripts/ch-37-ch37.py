#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — شجرة القرار: من سؤال إلى شجرة
Mastering scikit-learn 1.9 — Chapter 37 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-37-ch37 (lesson 37)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-37-ch37.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — شجرة القرار: من سؤال إلى شجرة
Mastering scikit-learn 1.9 — Chapter 37 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-37-ch37 (lesson 37)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                           n_redundant=0, class_sep=2.2, random_state=42)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["label"] = y

print(f"shape         : {df.shape}")
print(f"class balance : {np.bincount(y)}  (0 = class A, 1 = class B)")
print()
print(df.head(8).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
root_probs = np.bincount(y) / len(y)


def gini_of(labels):
    counts = np.bincount(labels, minlength=2).astype(float)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def entropy_of(labels):
    counts = np.bincount(labels, minlength=2).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


print(f"class proportions       : {np.round(root_probs, 4)}")
print(f"Gini of the whole set   : {gini_of(y):.4f}")
print(f"Entropy of the whole set: {entropy_of(y):.4f} bits")
print(f"hand check: 1 - sum(p^2): {1 - np.sum(root_probs ** 2):.4f}")
print(f"maximum entropy, 2 classes: {np.log2(2):.4f} bits")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
def weighted_gini(labels, mask):
    left, right = labels[mask], labels[~mask]
    n = len(labels)
    return (len(left) / n) * gini_of(left) + (len(right) / n) * gini_of(right)


root_gini = gini_of(y)
print(f"root Gini = {root_gini:.4f}\n")
for j, name in enumerate(["x1", "x2", "x3"]):
    m = np.median(X[:, j])
    mask = X[:, j] <= m
    w = weighted_gini(y, mask)
    print(f"split on {name} <= {m:7.3f} : weighted Gini = {w:.4f}"
          f"   gain = {root_gini - w:+.4f}   left n = {mask.sum()}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
best_overall = []

for j, name in enumerate(["x1", "x2", "x3"]):
    gains = []
    for v in np.unique(X[:, j])[:-1]:
        mask = X[:, j] <= v
        if mask.all() or (~mask).all():
            continue
        gains.append((root_gini - weighted_gini(y, mask), v, int(mask.sum())))
    g, v, n_left = max(gains)
    best_overall.append((g, name, v))
    print(f"{name}: best gain {g:.4f} at threshold {v:8.4f}  (left n = {n_left})")

gain, best_feature, best_threshold = max(best_overall)
print(f"\n-> the tree splits the root on {best_feature} <= {best_threshold:.4f}")
print(f"-> it prefers {best_feature} because its gain ({gain:.4f}) is the largest")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42).fit(X, y)

print(f"tree depth        : {clf.get_depth()}")
print(f"number of leaves  : {clf.get_n_leaves()}")
print(f"features seen     : {clf.n_features_in_}")
print(f"root feature      : x{clf.tree_.feature[0]}")
print(f"root threshold    : {clf.tree_.threshold[0]:.4f}")
print()
imp = pd.Series(clf.feature_importances_, index=["x1", "x2", "x3"])
print("feature_importances_:")
print(imp.to_frame("importance").to_string())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part05/ch37.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer

Xb, yb = load_breast_cancer(return_X_y=True)
Xb_holes = Xb.copy()
rng = np.random.default_rng(0)
holes = rng.random(Xb.shape) < 0.06
Xb_holes[holes] = np.nan

gapped = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb_holes, yb)

print(f"features in            : {gapped.n_features_in_}")
print(f"missing cells fed in   : {int(holes.sum())}")
print(f"accuracy on gapped set : {gapped.score(Xb_holes, yb):.4f}")
print(f"nodes routing on NaN   : {int(gapped.tree_.missing_go_to_left.sum())}")
print()
print("NaN is not an error here. It is a third option: <= v, > v, or is-missing.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                           n_redundant=0, class_sep=2.2, random_state=42)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["label"] = y

print(f"shape         : {df.shape}")
print(f"class balance : {np.bincount(y)}  (0 = class A, 1 = class B)")
print()
print(df.head(8).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
root_probs = np.bincount(y) / len(y)


def gini_of(labels):
    counts = np.bincount(labels, minlength=2).astype(float)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def entropy_of(labels):
    counts = np.bincount(labels, minlength=2).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


print(f"class proportions       : {np.round(root_probs, 4)}")
print(f"Gini of the whole set   : {gini_of(y):.4f}")
print(f"Entropy of the whole set: {entropy_of(y):.4f} bits")
print(f"hand check: 1 - sum(p^2): {1 - np.sum(root_probs ** 2):.4f}")
print(f"maximum entropy, 2 classes: {np.log2(2):.4f} bits")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def weighted_gini(labels, mask):
    left, right = labels[mask], labels[~mask]
    n = len(labels)
    return (len(left) / n) * gini_of(left) + (len(right) / n) * gini_of(right)


root_gini = gini_of(y)
print(f"root Gini = {root_gini:.4f}\n")
for j, name in enumerate(["x1", "x2", "x3"]):
    m = np.median(X[:, j])
    mask = X[:, j] <= m
    w = weighted_gini(y, mask)
    print(f"split on {name} <= {m:7.3f} : weighted Gini = {w:.4f}"
          f"   gain = {root_gini - w:+.4f}   left n = {mask.sum()}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
best_overall = []

for j, name in enumerate(["x1", "x2", "x3"]):
    gains = []
    for v in np.unique(X[:, j])[:-1]:
        mask = X[:, j] <= v
        if mask.all() or (~mask).all():
            continue
        gains.append((root_gini - weighted_gini(y, mask), v, int(mask.sum())))
    g, v, n_left = max(gains)
    best_overall.append((g, name, v))
    print(f"{name}: best gain {g:.4f} at threshold {v:8.4f}  (left n = {n_left})")

gain, best_feature, best_threshold = max(best_overall)
print(f"\n-> the tree splits the root on {best_feature} <= {best_threshold:.4f}")
print(f"-> it prefers {best_feature} because its gain ({gain:.4f}) is the largest")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42).fit(X, y)

print(f"tree depth        : {clf.get_depth()}")
print(f"number of leaves  : {clf.get_n_leaves()}")
print(f"features seen     : {clf.n_features_in_}")
print(f"root feature      : x{clf.tree_.feature[0]}")
print(f"root threshold    : {clf.tree_.threshold[0]:.4f}")
print()
imp = pd.Series(clf.feature_importances_, index=["x1", "x2", "x3"])
print("feature_importances_:")
print(imp.to_frame("importance").to_string())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-37-ch37.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer

Xb, yb = load_breast_cancer(return_X_y=True)
Xb_holes = Xb.copy()
rng = np.random.default_rng(0)
holes = rng.random(Xb.shape) < 0.06
Xb_holes[holes] = np.nan

gapped = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb_holes, yb)

print(f"features in            : {gapped.n_features_in_}")
print(f"missing cells fed in   : {int(holes.sum())}")
print(f"accuracy on gapped set : {gapped.score(Xb_holes, yb):.4f}")
print(f"nodes routing on NaN   : {int(gapped.tree_.missing_go_to_left.sum())}")
print()
print("NaN is not an error here. It is a third option: <= v, > v, or is-missing.")
