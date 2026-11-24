#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التقاط التقييمات وتتبّع الأدوات
Mastering scikit-learn 1.9 — Chapter 74 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-74-ch74 (lesson 74)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-74-ch74.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التقاط التقييمات وتتبّع الأدوات
Mastering scikit-learn 1.9 — Chapter 74 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-74-ch74 (lesson 74)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
details = cross_validate(model, X, y, cv=5,
                         scoring=["accuracy", "balanced_accuracy"],
                         return_train_score=True)
print("درجات balanced accuracy:", scores.round(3).tolist())
print("مفاتيح التفاصيل:", sorted(details))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
scoring = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "recall_macro": make_scorer(recall_score, average="macro"),
}
result = cross_validate(model, X, y, cv=5, scoring=scoring,
                        return_train_score=True)
for key in ("test_f1_macro", "test_recall_macro", "fit_time", "score_time"):
    values = result[key]
    print(f"{key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
pred = cross_val_predict(model, X, y, cv=cv)
print("عدد التنبؤات:", len(pred), "| عدد العينات:", len(y))
print("مصفوفة الخلط المجمّعة:\n", confusion_matrix(y, pred))
print("دقة التنبؤات المجمّعة:", round(np.mean(pred == y), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
result = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy",
                        n_jobs=2, return_train_score=True,
                        error_score="raise")
print("درجات الاختبار:", result["test_score"].round(3).tolist())
print("مجموع زمن الملاءمة:", round(result["fit_time"].sum(), 3), "ثانية")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
r = cross_validate(model, X, y, cv=5,
                   scoring=["accuracy", "balanced_accuracy"],
                   return_train_score=True)
for k in ("train_accuracy", "test_accuracy",
          "train_balanced_accuracy", "test_balanced_accuracy"):
    print(k, round(float(np.mean(r[k])), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch74.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
p = cross_val_predict(make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000)), X, y, cv=cv)
assert p.shape[0] == X.shape[0]
print("التغطية كاملة:", p.shape[0])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
details = cross_validate(model, X, y, cv=5,
                         scoring=["accuracy", "balanced_accuracy"],
                         return_train_score=True)
print("درجات balanced accuracy:", scores.round(3).tolist())
print("مفاتيح التفاصيل:", sorted(details))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
scoring = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "recall_macro": make_scorer(recall_score, average="macro"),
}
result = cross_validate(model, X, y, cv=5, scoring=scoring,
                        return_train_score=True)
for key in ("test_f1_macro", "test_recall_macro", "fit_time", "score_time"):
    values = result[key]
    print(f"{key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
pred = cross_val_predict(model, X, y, cv=cv)
print("عدد التنبؤات:", len(pred), "| عدد العينات:", len(y))
print("مصفوفة الخلط المجمّعة:\n", confusion_matrix(y, pred))
print("دقة التنبؤات المجمّعة:", round(np.mean(pred == y), 4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
result = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy",
                        n_jobs=2, return_train_score=True,
                        error_score="raise")
print("درجات الاختبار:", result["test_score"].round(3).tolist())
print("مجموع زمن الملاءمة:", round(result["fit_time"].sum(), 3), "ثانية")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
r = cross_validate(model, X, y, cv=5,
                   scoring=["accuracy", "balanced_accuracy"],
                   return_train_score=True)
for k in ("train_accuracy", "test_accuracy",
          "train_balanced_accuracy", "test_balanced_accuracy"):
    print(k, round(float(np.mean(r[k])), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-74-ch74.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
p = cross_val_predict(make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000)), X, y, cv=cv)
assert p.shape[0] == X.shape[0]
print("التغطية كاملة:", p.shape[0])
