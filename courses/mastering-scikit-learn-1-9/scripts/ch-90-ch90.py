#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهة الاستدعاءات الجديدة
Mastering scikit-learn 1.9 — Chapter 90 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-90-ch90 (lesson 90)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-90-ch90.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهة الاستدعاءات الجديدة
Mastering scikit-learn 1.9 — Chapter 90 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-90-ch90 (lesson 90)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar, ScoringMonitor
from sklearn.linear_model import LogisticRegression, LinearRegression

for estimator in (LogisticRegression(), LinearRegression()):
    print(type(estimator).__name__,
          "set_callbacks:", hasattr(estimator, "set_callbacks"))
print("كائنات callback:",
      type(ProgressBar()).__name__,
      type(ScoringMonitor(scoring="accuracy")).__name__)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, random_state=42)
X = StandardScaler().fit_transform(X)
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.set_callbacks(ProgressBar())
model.fit(X, y)
print("عدد التكرارات المسجل:", model.n_iter_.tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=600, n_features=16,
                           n_informative=6, random_state=42)
X = StandardScaler().fit_transform(X)
monitor = ScoringMonitor(scoring="accuracy")
model = LogisticRegression(max_iter=300, solver="lbfgs")
model.set_callbacks(monitor)
model.fit(X, y)
logs = monitor.get_logs()
print("نوع السجل:", type(logs).__name__)
print("عدد الصفوف:", logs.data_as_pandas.shape[0])
print("الأعمدة:", logs.data_as_pandas.columns.tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, solver="lbfgs"))
scores = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy")
print("درجات التحقق:", scores["test_score"].round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

for estimator in (LogisticRegression(), DecisionTreeClassifier()):
    print(type(estimator).__name__, hasattr(estimator, "set_callbacks"))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch90.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=700, n_features=16,
                           n_informative=6, random_state=42)
X = StandardScaler().fit_transform(X)
for limit in (10, 500):
    clf = LogisticRegression(max_iter=limit, solver="lbfgs")
    clf.set_callbacks(ProgressBar())
    clf.fit(X, y)
    print("max_iter:", limit, "n_iter_:", clf.n_iter_.tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar, ScoringMonitor
from sklearn.linear_model import LogisticRegression, LinearRegression

for estimator in (LogisticRegression(), LinearRegression()):
    print(type(estimator).__name__,
          "set_callbacks:", hasattr(estimator, "set_callbacks"))
print("كائنات callback:",
      type(ProgressBar()).__name__,
      type(ScoringMonitor(scoring="accuracy")).__name__)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, random_state=42)
X = StandardScaler().fit_transform(X)
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.set_callbacks(ProgressBar())
model.fit(X, y)
print("عدد التكرارات المسجل:", model.n_iter_.tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=600, n_features=16,
                           n_informative=6, random_state=42)
X = StandardScaler().fit_transform(X)
monitor = ScoringMonitor(scoring="accuracy")
model = LogisticRegression(max_iter=300, solver="lbfgs")
model.set_callbacks(monitor)
model.fit(X, y)
logs = monitor.get_logs()
print("نوع السجل:", type(logs).__name__)
print("عدد الصفوف:", logs.data_as_pandas.shape[0])
print("الأعمدة:", logs.data_as_pandas.columns.tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, solver="lbfgs"))
scores = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy")
print("درجات التحقق:", scores["test_score"].round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

for estimator in (LogisticRegression(), DecisionTreeClassifier()):
    print(type(estimator).__name__, hasattr(estimator, "set_callbacks"))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-90-ch90.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.callback import ProgressBar
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=700, n_features=16,
                           n_informative=6, random_state=42)
X = StandardScaler().fit_transform(X)
for limit in (10, 500):
    clf = LogisticRegression(max_iter=limit, solver="lbfgs")
    clf.set_callbacks(ProgressBar())
    clf.fit(X, y)
    print("max_iter:", limit, "n_iter_:", clf.n_iter_.tolist())
