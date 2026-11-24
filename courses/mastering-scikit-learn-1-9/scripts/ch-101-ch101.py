#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٣: كشف عمليات الاحتيال
Mastering scikit-learn 1.9 — Chapter 101 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-101-ch101 (lesson 101)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-101-ch101.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٣: كشف عمليات الاحتيال
Mastering scikit-learn 1.9 — Chapter 101 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-101-ch101 (lesson 101)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=5000, n_features=24, n_informative=7, n_redundant=4,
    weights=[0.985, 0.015], class_sep=0.6, flip_y=0.01, random_state=42)
print("المعاملات:", X.shape[0])
print("حالات الفئة النادرة:", int(y.sum()))
print("انتشار الإيجابي:", round(float(y.mean()), 4))
print("خط أساس دائماً سليم:", round(float((y == 0).mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

X, y = make_classification(
    n_samples=5000, n_features=24, n_informative=7, n_redundant=4,
    weights=[0.985, 0.015], class_sep=0.6, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]
print("Average precision:", round(average_precision_score(yte, prob), 4))
print("ROC AUC:", round(roc_auc_score(yte, prob), 4))
print("انتشار الاختبار:", round(float(yte.mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=4000, n_features=20, n_informative=7, weights=[0.985, 0.015],
    class_sep=0.65, random_state=17)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=17)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced"))
m.fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]
for threshold in (0.35, 0.65):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"t={threshold:.2f}: review={int(pred.sum())}, TP={tp}, FP={fp}, FN={fn}, "
          f"precision={precision_score(yte, pred, zero_division=0):.3f}, "
          f"recall={recall_score(yte, pred):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=3000, n_features=18, n_informative=6,
                           weights=[0.98, 0.02], random_state=5)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=5)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced")).fit(a, c)
p = m.predict_proba(b)[:, 1]
precision, recall, thresholds = precision_recall_curve(d, p)
print("AP:", round(average_precision_score(d, p), 4))
print("منحنى PR:", len(precision), len(recall), len(thresholds))
print("عتبات مرشحة:", np.round(np.quantile(thresholds, [0.25, 0.5, 0.75]), 3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=10, weights=[0.99, 0.01],
                           random_state=42)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
m = DummyClassifier(strategy="most_frequent").fit(a, c)
p = m.predict(b)
print("accuracy:", round(accuracy_score(d, p), 4))
print("average precision:", round(average_precision_score(d, p), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch101.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

scores = np.array([0.05, 0.12, 0.22, 0.31, 0.48, 0.61, 0.8])
for threshold in (0.25, 0.5):
    print(threshold, "عدد المراجعات:", int((scores >= threshold).sum()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=5000, n_features=24, n_informative=7, n_redundant=4,
    weights=[0.985, 0.015], class_sep=0.6, flip_y=0.01, random_state=42)
print("المعاملات:", X.shape[0])
print("حالات الفئة النادرة:", int(y.sum()))
print("انتشار الإيجابي:", round(float(y.mean()), 4))
print("خط أساس دائماً سليم:", round(float((y == 0).mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

X, y = make_classification(
    n_samples=5000, n_features=24, n_informative=7, n_redundant=4,
    weights=[0.985, 0.015], class_sep=0.6, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]
print("Average precision:", round(average_precision_score(yte, prob), 4))
print("ROC AUC:", round(roc_auc_score(yte, prob), 4))
print("انتشار الاختبار:", round(float(yte.mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=4000, n_features=20, n_informative=7, weights=[0.985, 0.015],
    class_sep=0.65, random_state=17)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=17)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced"))
m.fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]
for threshold in (0.35, 0.65):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"t={threshold:.2f}: review={int(pred.sum())}, TP={tp}, FP={fp}, FN={fn}, "
          f"precision={precision_score(yte, pred, zero_division=0):.3f}, "
          f"recall={recall_score(yte, pred):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=3000, n_features=18, n_informative=6,
                           weights=[0.98, 0.02], random_state=5)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=5)
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced")).fit(a, c)
p = m.predict_proba(b)[:, 1]
precision, recall, thresholds = precision_recall_curve(d, p)
print("AP:", round(average_precision_score(d, p), 4))
print("منحنى PR:", len(precision), len(recall), len(thresholds))
print("عتبات مرشحة:", np.round(np.quantile(thresholds, [0.25, 0.5, 0.75]), 3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=10, weights=[0.99, 0.01],
                           random_state=42)
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
m = DummyClassifier(strategy="most_frequent").fit(a, c)
p = m.predict(b)
print("accuracy:", round(accuracy_score(d, p), 4))
print("average precision:", round(average_precision_score(d, p), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-101-ch101.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

scores = np.array([0.05, 0.12, 0.22, 0.31, 0.48, 0.61, 0.8])
for threshold in (0.25, 0.5):
    print(threshold, "عدد المراجعات:", int((scores >= threshold).sum()))
