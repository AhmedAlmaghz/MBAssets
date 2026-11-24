#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهات العرض البياني في 1.9
Mastering scikit-learn 1.9 — Chapter 94 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-94-ch94 (lesson 94)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-94-ch94.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهات العرض البياني في 1.9
Mastering scikit-learn 1.9 — Chapter 94 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-94-ch94 (lesson 94)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
model.fit(Xtr, ytr)
display = ConfusionMatrixDisplay.from_estimator(model, Xte, yte)
print("كائن العرض:", type(display).__name__)
print("مصفوفة الخلط:", display.confusion_matrix.tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
model.fit(Xtr, ytr)
roc = RocCurveDisplay.from_estimator(model, Xte, yte)
print("AUC:", round(roc.roc_auc, 4))
print("نقاط المنحنى:", len(roc.fpr))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtr, ytr)
features = [X.columns[0], X.columns[1]]
display = PartialDependenceDisplay.from_estimator(
    model, Xte, features=features, kind="average", grid_resolution=20)
print("الخصائص المعروضة:", display.feature_names)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_html_repr

set_config(display="diagram")
model = make_pipeline(StandardScaler(),
                      LogisticRegression(C=0.5, max_iter=1000))
html = estimator_html_repr(model)
print("وسم HTML:", html[:40])
print("اسم الخطوة موجود:", "StandardScaler" in html)
print("طول التمثيل:", len(html))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ConfusionMatrixDisplay

y_true = [0, 0, 1, 1, 1]
y_pred = [0, 1, 1, 0, 1]
d = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1])
print(d.confusion_matrix.tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch94.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(a, c)
r = RocCurveDisplay.from_estimator(m, b, d)
print(round(r.roc_auc, 4), len(r.fpr))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
model.fit(Xtr, ytr)
display = ConfusionMatrixDisplay.from_estimator(model, Xte, yte)
print("كائن العرض:", type(display).__name__)
print("مصفوفة الخلط:", display.confusion_matrix.tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
model.fit(Xtr, ytr)
roc = RocCurveDisplay.from_estimator(model, Xte, yte)
print("AUC:", round(roc.roc_auc, 4))
print("نقاط المنحنى:", len(roc.fpr))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtr, ytr)
features = [X.columns[0], X.columns[1]]
display = PartialDependenceDisplay.from_estimator(
    model, Xte, features=features, kind="average", grid_resolution=20)
print("الخصائص المعروضة:", display.feature_names)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_html_repr

set_config(display="diagram")
model = make_pipeline(StandardScaler(),
                      LogisticRegression(C=0.5, max_iter=1000))
html = estimator_html_repr(model)
print("وسم HTML:", html[:40])
print("اسم الخطوة موجود:", "StandardScaler" in html)
print("طول التمثيل:", len(html))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ConfusionMatrixDisplay

y_true = [0, 0, 1, 1, 1]
y_pred = [0, 1, 1, 0, 1]
d = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1])
print(d.confusion_matrix.tolist())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-94-ch94.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(a, c)
r = RocCurveDisplay.from_estimator(m, b, d)
print(round(r.roc_auc, 4), len(r.fpr))
