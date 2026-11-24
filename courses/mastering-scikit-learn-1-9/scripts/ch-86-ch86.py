#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — FeatureUnion والتركيبات المتقدمة
Mastering scikit-learn 1.9 — FeatureUnion

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-86-ch86 (lesson 86)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-86-ch86.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — FeatureUnion والتركيبات المتقدمة
Mastering scikit-learn 1.9 — FeatureUnion

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-86-ch86 (lesson 86)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import make_regression

X, _ = make_regression(n_samples=30, n_features=2, random_state=42)
union = FeatureUnion([
    ("scaled", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
])
Xt = union.fit_transform(X)
print("الأصل:", X.shape, "| الناتج:", Xt.shape)
print("الأسماء:", union.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
union = FeatureUnion([
    ("all_scaled", StandardScaler()),
    ("all_raw", FunctionTransformer(validate=False)),
])
columns = ColumnTransformer([
    ("first", StandardScaler(), [0]),
    ("second_raw", "passthrough", [1]),
])
print("FeatureUnion:", union.fit_transform(X).shape)
print("ColumnTransformer:", columns.fit_transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
features = FeatureUnion([
    ("scaled", StandardScaler()),
    ("selected", SelectKBest(f_classif, k=10)),
])
pipe = Pipeline([("features", features),
                 ("clf", LogisticRegression(max_iter=3000))])
print("معامل فرع الاختيار:", pipe.get_params()["features__selected__k"])
pipe.fit(X, y)
print("شكل الخصائص المدمجة:",
      pipe.named_steps["features"].transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=240, n_features=12, n_informative=5,
                           random_state=42)
union = FeatureUnion([("scale", StandardScaler()),
                      ("select", SelectKBest(f_classif, k=5))])
model = Pipeline([("union", union),
                  ("clf", LogisticRegression(max_iter=1000))])
scores = cross_val_score(model, X, y, cv=4, scoring="accuracy")
print("درجات الطيات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

X = np.arange(18, dtype=float).reshape(6, 3)
u = FeatureUnion([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
])
print("الأشكال:", X.shape, u.fit_transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part10/ch86.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

X = np.arange(12, dtype=float).reshape(6, 2)
ct = ColumnTransformer([("left", StandardScaler(), [0]),
                        ("right", "passthrough", [1])])
fu = FeatureUnion([("scaled", StandardScaler()),
                   ("raw", "passthrough")])
print("الأعمدة:", ct.fit_transform(X).shape, fu.fit_transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import make_regression

X, _ = make_regression(n_samples=30, n_features=2, random_state=42)
union = FeatureUnion([
    ("scaled", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
])
Xt = union.fit_transform(X)
print("الأصل:", X.shape, "| الناتج:", Xt.shape)
print("الأسماء:", union.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
union = FeatureUnion([
    ("all_scaled", StandardScaler()),
    ("all_raw", FunctionTransformer(validate=False)),
])
columns = ColumnTransformer([
    ("first", StandardScaler(), [0]),
    ("second_raw", "passthrough", [1]),
])
print("FeatureUnion:", union.fit_transform(X).shape)
print("ColumnTransformer:", columns.fit_transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
features = FeatureUnion([
    ("scaled", StandardScaler()),
    ("selected", SelectKBest(f_classif, k=10)),
])
pipe = Pipeline([("features", features),
                 ("clf", LogisticRegression(max_iter=3000))])
print("معامل فرع الاختيار:", pipe.get_params()["features__selected__k"])
pipe.fit(X, y)
print("شكل الخصائص المدمجة:",
      pipe.named_steps["features"].transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=240, n_features=12, n_informative=5,
                           random_state=42)
union = FeatureUnion([("scale", StandardScaler()),
                      ("select", SelectKBest(f_classif, k=5))])
model = Pipeline([("union", union),
                  ("clf", LogisticRegression(max_iter=1000))])
scores = cross_val_score(model, X, y, cv=4, scoring="accuracy")
print("درجات الطيات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

X = np.arange(18, dtype=float).reshape(6, 3)
u = FeatureUnion([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
])
print("الأشكال:", X.shape, u.fit_transform(X).shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-86-ch86.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

X = np.arange(12, dtype=float).reshape(6, 2)
ct = ColumnTransformer([("left", StandardScaler(), [0]),
                        ("right", "passthrough", [1])])
fu = FeatureUnion([("scaled", StandardScaler()),
                   ("raw", "passthrough")])
print("الأعمدة:", ct.fit_transform(X).shape, fu.fit_transform(X).shape)
