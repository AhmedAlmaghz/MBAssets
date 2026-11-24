#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التوسّع والتخصيص
Mastering scikit-learn 1.9 — Chapter 96 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-96-ch96 (lesson 96)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-96-ch96.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التوسّع والتخصيص
Mastering scikit-learn 1.9 — Chapter 96 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-96-ch96 (lesson 96)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class MeanCenterer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="numeric")
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        X = validate_data(self, X, reset=False, dtype="numeric")
        return X - self.mean_

X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
transformer = MeanCenterer().fit(X)
print("المتوسط المخزن:", transformer.mean_.tolist())
print("المخرجات:", transformer.transform(X).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

center = MeanCenterer()
print("المعاملات:", center.get_params(deep=False))
print("الاستنساخ جديد:", clone(center) is not center)
pipe = Pipeline([("scale", StandardScaler()), ("center", MeanCenterer())])
pipe.fit([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]])
print("أسماء الخطوات:", list(pipe.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.estimator_checks import check_estimator

# انقل MeanCenterer إلى ملف Python داخل مشروعك ثم استورده هنا.
check_estimator(MeanCenterer())

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = Pipeline([
    ("scale", StandardScaler()),
    ("center", MeanCenterer()),
    ("clf", LogisticRegression(max_iter=3000)),
])
scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
print("الدرجات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin, clone

class ConfigurableCenterer(TransformerMixin, BaseEstimator):
    def __init__(self, scale=False):
        self.scale = scale

model = ConfigurableCenterer(scale=True)
print(model.get_params(deep=False))
print(clone(model).get_params(deep=False))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch96.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import StandardScaler

with config_context(transform_output="default"):
    check_estimator(StandardScaler())
print("اجتازت الفحوص العامة.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class MeanCenterer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="numeric")
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        X = validate_data(self, X, reset=False, dtype="numeric")
        return X - self.mean_

X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
transformer = MeanCenterer().fit(X)
print("المتوسط المخزن:", transformer.mean_.tolist())
print("المخرجات:", transformer.transform(X).tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

center = MeanCenterer()
print("المعاملات:", center.get_params(deep=False))
print("الاستنساخ جديد:", clone(center) is not center)
pipe = Pipeline([("scale", StandardScaler()), ("center", MeanCenterer())])
pipe.fit([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]])
print("أسماء الخطوات:", list(pipe.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.estimator_checks import check_estimator

# انقل MeanCenterer إلى ملف Python داخل مشروعك ثم استورده هنا.
check_estimator(MeanCenterer())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = Pipeline([
    ("scale", StandardScaler()),
    ("center", MeanCenterer()),
    ("clf", LogisticRegression(max_iter=3000)),
])
scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
print("الدرجات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin, clone

class ConfigurableCenterer(TransformerMixin, BaseEstimator):
    def __init__(self, scale=False):
        self.scale = scale

model = ConfigurableCenterer(scale=True)
print(model.get_params(deep=False))
print(clone(model).get_params(deep=False))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-96-ch96.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import StandardScaler

with config_context(transform_output="default"):
    check_estimator(StandardScaler())
print("اجتازت الفحوص العامة.")
