#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحقق المتقاطع: كل أنواعه
Mastering scikit-learn 1.9 — Chapter 73 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-73-ch73 (lesson 73)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-73-ch73.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحقق المتقاطع: كل أنواعه
Mastering scikit-learn 1.9 — Chapter 73 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-73-ch73 (lesson 73)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(12).reshape(-1, 1)
cv = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
    print(f"الطية {fold}: تدريب={train_idx.tolist()} اختبار={test_idx.tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import StratifiedKFold

y = np.array([0] * 18 + [1] * 6)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for fold, (_, test_idx) in enumerate(cv.split(np.zeros((len(y), 1)), y), 1):
    labels, counts = np.unique(y[test_idx], return_counts=True)
    print(f"الطية {fold}: {dict(zip(labels.tolist(), counts.tolist()))}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(6), 3)
X = np.arange(len(groups)).reshape(-1, 1)
cv = GroupKFold(n_splits=3)
for fold, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups), 1):
    overlap = np.intersect1d(groups[train_idx], groups[test_idx])
    print(f"الطية {fold}: مجموعات الاختبار={np.unique(groups[test_idx]).tolist()}, "
          f"تداخل={overlap.tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(20).reshape(-1, 1)
cv = TimeSeriesSplit(n_splits=4, test_size=2, gap=1)
for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    print(f"الطية {fold}: آخر تدريب={train_idx[-1]}, "
          f"أول اختبار={test_idx[0]}, حجم الاختبار={len(test_idx)}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

y = np.array([0] * 15 + [1] * 6 + [2] * 3)
X = np.zeros((len(y), 1))
for name, cv in [
    ("KFold", KFold(3, shuffle=True, random_state=42)),
    ("StratifiedKFold", StratifiedKFold(3, shuffle=True, random_state=42)),
]:
    print(name)
    for _, idx in cv.split(X, y):
        print(np.bincount(y[idx], minlength=3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch73.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(12), 4)
X = np.zeros((len(groups), 1))
for train_idx, test_idx in GroupKFold(4).split(X, groups=groups):
    assert not np.intersect1d(groups[train_idx], groups[test_idx]).size
print("نجحت فحوص الطيات الأربع: لا مجموعة مشتركة.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(12).reshape(-1, 1)
cv = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
    print(f"الطية {fold}: تدريب={train_idx.tolist()} اختبار={test_idx.tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import StratifiedKFold

y = np.array([0] * 18 + [1] * 6)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for fold, (_, test_idx) in enumerate(cv.split(np.zeros((len(y), 1)), y), 1):
    labels, counts = np.unique(y[test_idx], return_counts=True)
    print(f"الطية {fold}: {dict(zip(labels.tolist(), counts.tolist()))}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(6), 3)
X = np.arange(len(groups)).reshape(-1, 1)
cv = GroupKFold(n_splits=3)
for fold, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups), 1):
    overlap = np.intersect1d(groups[train_idx], groups[test_idx])
    print(f"الطية {fold}: مجموعات الاختبار={np.unique(groups[test_idx]).tolist()}, "
          f"تداخل={overlap.tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(20).reshape(-1, 1)
cv = TimeSeriesSplit(n_splits=4, test_size=2, gap=1)
for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    print(f"الطية {fold}: آخر تدريب={train_idx[-1]}, "
          f"أول اختبار={test_idx[0]}, حجم الاختبار={len(test_idx)}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

y = np.array([0] * 15 + [1] * 6 + [2] * 3)
X = np.zeros((len(y), 1))
for name, cv in [
    ("KFold", KFold(3, shuffle=True, random_state=42)),
    ("StratifiedKFold", StratifiedKFold(3, shuffle=True, random_state=42)),
]:
    print(name)
    for _, idx in cv.split(X, y):
        print(np.bincount(y[idx], minlength=3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-73-ch73.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(12), 4)
X = np.zeros((len(groups), 1))
for train_idx, test_idx in GroupKFold(4).split(X, groups=groups):
    assert not np.intersect1d(groups[train_idx], groups[test_idx]).size
print("نجحت فحوص الطيات الأربع: لا مجموعة مشتركة.")
