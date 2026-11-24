#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — معجم الأخطاء: من الرسالة إلى الحل
Mastering scikit-learn 1.9 — Chapter 98 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-98-ch98 (lesson 98)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-98-ch98.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — معجم الأخطاء: من الرسالة إلى الحل
Mastering scikit-learn 1.9 — Chapter 98 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-98-ch98 (lesson 98)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([0, 1])
try:
    LogisticRegression().fit(X, y)
except ValueError as exc:
    print(type(exc).__name__)
    print(str(exc).splitlines()[0])
print("X:", X.shape, "| y:", y.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]])
y = np.array([0, 0, 1, 1])
print("عدد المفقودات لكل عمود:", np.isnan(X).sum(axis=0).tolist())
model = make_pipeline(SimpleImputer(strategy="median"),
                      LogisticRegression())
model.fit(X, y)
print("عدد التنبؤات:", model.predict(X).shape[0])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
try:
    model.predict([[1.0, 2.0]])
except NotFittedError as exc:
    print("قبل fit:", type(exc).__name__)
model.fit([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
          [0, 0, 1, 1])
print("بعد fit:", model.predict([[1.5, 0.5]]).tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
import warnings
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=20,
                           n_informative=10, random_state=42)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = LogisticRegression(max_iter=1).fit(X, y)
print("التكرارات:", model.n_iter_.tolist())
print("التحذيرات:", [type(w.message).__name__ for w in caught])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.ones((5, 2))
y = np.array([0, 1, 0, 1])
if X.shape[0] != y.shape[0]:
    print(f"عدد الصفوف مختلف: X={X.shape[0]}, y={y.shape[0]}")
else:
    LogisticRegression().fit(X, y)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch98.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]])
y = [0, 0, 1, 1]
model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression())
model.fit(X, y)
print(model.predict(X).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([0, 1])
try:
    LogisticRegression().fit(X, y)
except ValueError as exc:
    print(type(exc).__name__)
    print(str(exc).splitlines()[0])
print("X:", X.shape, "| y:", y.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]])
y = np.array([0, 0, 1, 1])
print("عدد المفقودات لكل عمود:", np.isnan(X).sum(axis=0).tolist())
model = make_pipeline(SimpleImputer(strategy="median"),
                      LogisticRegression())
model.fit(X, y)
print("عدد التنبؤات:", model.predict(X).shape[0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
try:
    model.predict([[1.0, 2.0]])
except NotFittedError as exc:
    print("قبل fit:", type(exc).__name__)
model.fit([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
          [0, 0, 1, 1])
print("بعد fit:", model.predict([[1.5, 0.5]]).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import warnings
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=20,
                           n_informative=10, random_state=42)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = LogisticRegression(max_iter=1).fit(X, y)
print("التكرارات:", model.n_iter_.tolist())
print("التحذيرات:", [type(w.message).__name__ for w in caught])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.ones((5, 2))
y = np.array([0, 1, 0, 1])
if X.shape[0] != y.shape[0]:
    print(f"عدد الصفوف مختلف: X={X.shape[0]}, y={y.shape[0]}")
else:
    LogisticRegression().fit(X, y)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-98-ch98.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]])
y = [0, 0, 1, 1]
model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression())
model.fit(X, y)
print(model.predict(X).tolist())
