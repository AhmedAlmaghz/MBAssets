#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٢: توقّع مغادرة العملاء
Mastering scikit-learn 1.9 — Chapter 100 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-100-ch100 (lesson 100)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-100-ch100.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٢: توقّع مغادرة العملاء
Mastering scikit-learn 1.9 — Chapter 100 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-100-ch100 (lesson 100)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

Xn, y = make_classification(n_samples=1200, n_features=3, n_informative=3,
                            n_redundant=0, weights=[0.78, 0.22],
                            class_sep=0.8, random_state=42)
rng = np.random.default_rng(42)
customers = pd.DataFrame({
    "tenure_months": np.clip((Xn[:, 0] * 12 + 30).round(), 1, 120),
    "monthly_charge": np.clip(Xn[:, 1] * 18 + 65, 10, 180).round(2),
    "support_calls": np.clip((Xn[:, 2] * 2 + 3).round(), 0, 15).astype(int),
    "contract": np.where(rng.random(len(y)) > 0.5, "annual", "monthly"),
    "churn": y,
})
print("الأبعاد:", customers.shape)
print("نسبة المغادرة:", round(customers["churn"].mean(), 3))
print(customers.head(3).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Xn, y = __import__("sklearn.datasets", fromlist=["make_classification"]).make_classification(
    n_samples=1000, n_features=3, n_informative=3, n_redundant=0,
    weights=[0.8, 0.2], random_state=42)
rng = np.random.default_rng(1)
X = pd.DataFrame({"tenure": Xn[:, 0] * 10 + 24,
                  "charges": Xn[:, 1] * 20 + 70,
                  "calls": np.maximum(0, (Xn[:, 2] * 2 + 3).round()),
                  "contract": np.where(rng.random(len(y)) > 0.5, "annual", "monthly")})
a, b, c, d = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
prep = ColumnTransformer([("num", numeric, ["tenure", "charges", "calls"]),
                          ("cat", categorical, ["contract"])])
model = Pipeline([("prep", prep),
                  ("clf", LogisticRegression(max_iter=2000,
                                             class_weight="balanced"))])
model.fit(a, c)
print("Balanced accuracy:", round(balanced_accuracy_score(d, model.predict(b)), 3))
print("شكل التدريب:", a.shape, "نسبة المغادرة في الاختبار:", round(float(d.mean()), 3))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1400, n_features=8, n_informative=5,
                           weights=[0.8, 0.2], class_sep=0.8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
threshold = 0.35
pred = (p >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
value = tp * (120 - 10) - fp * 10
print(f"العتبة={threshold:.2f} | TP={tp} FP={fp} FN={fn} TN={tn}")
print("قيمة الحملة وفق الافتراضات:", value, "وحدة")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=8, n_informative=5,
                           weights=[0.82, 0.18], random_state=24)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=24)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced"))
m.fit(a, c)
p = m.predict_proba(b)[:, 1]
pred = (p >= 0.4).astype(int)
print("انتشار الهدف:", round(float(d.mean()), 3))
print("العملاء المستهدفون:", int(pred.sum()), "من", len(pred))
print("precision:", round(precision_score(d, pred, zero_division=0), 3))
print("recall:", round(recall_score(d, pred), 3))
print("average precision:", round(average_precision_score(d, p), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
decision_days = 90
feature_window = (decision_days - 180, decision_days)
label_window = (decision_days, decision_days + 30)
print("نافذة السمات:", feature_window)
print("نافذة الهدف:", label_window)
assert feature_window[1] <= label_window[0]

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch100.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix

y = np.array([0, 0, 0, 1, 1, 1])
p = np.array([0.1, 0.35, 0.8, 0.2, 0.55, 0.9])
for threshold in (0.3, 0.6):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(threshold, "TP/FP/FN/TN =", tp, fp, fn, tn,
          "القيمة =", tp * (120 - 10) - fp * 10)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

Xn, y = make_classification(n_samples=1200, n_features=3, n_informative=3,
                            n_redundant=0, weights=[0.78, 0.22],
                            class_sep=0.8, random_state=42)
rng = np.random.default_rng(42)
customers = pd.DataFrame({
    "tenure_months": np.clip((Xn[:, 0] * 12 + 30).round(), 1, 120),
    "monthly_charge": np.clip(Xn[:, 1] * 18 + 65, 10, 180).round(2),
    "support_calls": np.clip((Xn[:, 2] * 2 + 3).round(), 0, 15).astype(int),
    "contract": np.where(rng.random(len(y)) > 0.5, "annual", "monthly"),
    "churn": y,
})
print("الأبعاد:", customers.shape)
print("نسبة المغادرة:", round(customers["churn"].mean(), 3))
print(customers.head(3).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Xn, y = __import__("sklearn.datasets", fromlist=["make_classification"]).make_classification(
    n_samples=1000, n_features=3, n_informative=3, n_redundant=0,
    weights=[0.8, 0.2], random_state=42)
rng = np.random.default_rng(1)
X = pd.DataFrame({"tenure": Xn[:, 0] * 10 + 24,
                  "charges": Xn[:, 1] * 20 + 70,
                  "calls": np.maximum(0, (Xn[:, 2] * 2 + 3).round()),
                  "contract": np.where(rng.random(len(y)) > 0.5, "annual", "monthly")})
a, b, c, d = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
prep = ColumnTransformer([("num", numeric, ["tenure", "charges", "calls"]),
                          ("cat", categorical, ["contract"])])
model = Pipeline([("prep", prep),
                  ("clf", LogisticRegression(max_iter=2000,
                                             class_weight="balanced"))])
model.fit(a, c)
print("Balanced accuracy:", round(balanced_accuracy_score(d, model.predict(b)), 3))
print("شكل التدريب:", a.shape, "نسبة المغادرة في الاختبار:", round(float(d.mean()), 3))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1400, n_features=8, n_informative=5,
                           weights=[0.8, 0.2], class_sep=0.8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
threshold = 0.35
pred = (p >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
value = tp * (120 - 10) - fp * 10
print(f"العتبة={threshold:.2f} | TP={tp} FP={fp} FN={fn} TN={tn}")
print("قيمة الحملة وفق الافتراضات:", value, "وحدة")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=8, n_informative=5,
                           weights=[0.82, 0.18], random_state=24)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=24)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced"))
m.fit(a, c)
p = m.predict_proba(b)[:, 1]
pred = (p >= 0.4).astype(int)
print("انتشار الهدف:", round(float(d.mean()), 3))
print("العملاء المستهدفون:", int(pred.sum()), "من", len(pred))
print("precision:", round(precision_score(d, pred, zero_division=0), 3))
print("recall:", round(recall_score(d, pred), 3))
print("average precision:", round(average_precision_score(d, p), 3))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
decision_days = 90
feature_window = (decision_days - 180, decision_days)
label_window = (decision_days, decision_days + 30)
print("نافذة السمات:", feature_window)
print("نافذة الهدف:", label_window)
assert feature_window[1] <= label_window[0]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-100-ch100.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix

y = np.array([0, 0, 0, 1, 1, 1])
p = np.array([0.1, 0.35, 0.8, 0.2, 0.55, 0.9])
for threshold in (0.3, 0.6):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(threshold, "TP/FP/FN/TN =", tp, fp, fn, tn,
          "القيمة =", tp * (120 - 10) - fp * 10)
