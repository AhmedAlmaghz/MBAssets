#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — دعم واجهة Array API
Mastering scikit-learn 1.9 — Array

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-92-ch92 (lesson 92)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-92-ch92.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — دعم واجهة Array API
Mastering scikit-learn 1.9 — Array

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-92-ch92 (lesson 92)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils._tags import get_tags
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

for estimator in (StandardScaler(), PCA(), LogisticRegression(),
                  RandomForestClassifier()):
    tags = get_tags(estimator)
    print(type(estimator).__name__,
          "array_api_support=", tags.array_api_support)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
import os
import sklearn
from sklearn import get_config

print("SCIPY_ARRAY_API:", os.environ.get("SCIPY_ARRAY_API", "<غير مضبوط>"))
print("array_api_dispatch في الإعداد:",
      "array_api_dispatch" in get_config())
print("نسخة scikit-learn:", sklearn.__version__)

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, _ = make_regression(n_samples=40, n_features=4, random_state=42)
try:
    with config_context(array_api_dispatch=True):
        Xt = StandardScaler().fit_transform(X)
    print("نجح التحويل:", Xt.shape)
except RuntimeError as exc:
    print("البيئة غير مهيأة لمسار Array API:", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.arange(24, dtype=float).reshape(6, 4)
scaler = StandardScaler()
Xt = scaler.fit_transform(X)
print("نوع الإدخال:", type(X).__name__, "نوع المخرج:", type(Xt).__name__)
array = Xt.to_numpy() if hasattr(Xt, "to_numpy") else Xt
print("الشكل:", Xt.shape, "الحجم بالبايت:", array.nbytes)
print("المتوسط بعد التحجيم:", np.round(array.mean(axis=0), 8).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils._tags import get_tags
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

for est in (StandardScaler(), PCA(), RandomForestClassifier()):
    print(type(est).__name__, get_tags(est).array_api_support)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch92.md
# ────────────────────────────────────────────────────────────────────────
import os
print("SCIPY_ARRAY_API =", os.environ.get("SCIPY_ARRAY_API", "<غير مضبوط>"))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils._tags import get_tags
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

for estimator in (StandardScaler(), PCA(), LogisticRegression(),
                  RandomForestClassifier()):
    tags = get_tags(estimator)
    print(type(estimator).__name__,
          "array_api_support=", tags.array_api_support)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import sklearn
from sklearn import get_config

print("SCIPY_ARRAY_API:", os.environ.get("SCIPY_ARRAY_API", "<غير مضبوط>"))
print("array_api_dispatch في الإعداد:",
      "array_api_dispatch" in get_config())
print("نسخة scikit-learn:", sklearn.__version__)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, _ = make_regression(n_samples=40, n_features=4, random_state=42)
try:
    with config_context(array_api_dispatch=True):
        Xt = StandardScaler().fit_transform(X)
    print("نجح التحويل:", Xt.shape)
except RuntimeError as exc:
    print("البيئة غير مهيأة لمسار Array API:", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.arange(24, dtype=float).reshape(6, 4)
scaler = StandardScaler()
Xt = scaler.fit_transform(X)
print("نوع الإدخال:", type(X).__name__, "نوع المخرج:", type(Xt).__name__)
array = Xt.to_numpy() if hasattr(Xt, "to_numpy") else Xt
print("الشكل:", Xt.shape, "الحجم بالبايت:", array.nbytes)
print("المتوسط بعد التحجيم:", np.round(array.mean(axis=0), 8).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils._tags import get_tags
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

for est in (StandardScaler(), PCA(), RandomForestClassifier()):
    print(type(est).__name__, get_tags(est).array_api_support)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-92-ch92.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import os
print("SCIPY_ARRAY_API =", os.environ.get("SCIPY_ARRAY_API", "<غير مضبوط>"))
