#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٦: الصيانة التنبّؤية للآلات
Mastering scikit-learn 1.9 — Chapter 104 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-104-ch104 (lesson 104)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-104-ch104.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٦: الصيانة التنبّؤية للآلات
Mastering scikit-learn 1.9 — Chapter 104 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-104-ch104 (lesson 104)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500
t = np.arange(n)
degradation = t / n
vibration = 0.4 + 0.25 * degradation + rng.normal(0, 0.08, n)
temperature = 60 + 12 * degradation + rng.normal(0, 3, n)
pressure = 100 - 10 * degradation + rng.normal(0, 4, n)
risk = 1 / (1 + np.exp(-(-5 + 5 * degradation + 4 * (vibration - 0.5))))
failure_next_24h = (rng.random(n) < risk * 0.12).astype(int)
data = pd.DataFrame({"time": t, "vibration": vibration,
                     "temperature": temperature, "pressure": pressure,
                     "failure_next_24h": failure_next_24h})
print("شكل الجدول:", data.shape)
print("معدل الهدف:", round(float(data["failure_next_24h"].mean()), 3))
print(data.tail(3).round(2).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(17)
n = 1800
t = np.arange(n)
age = t / n
vibration = 0.35 + 0.3 * age + rng.normal(0, 0.1, n)
temperature = 55 + 18 * age + rng.normal(0, 4, n)
pressure = 105 - 13 * age + rng.normal(0, 5, n)
prob = 1 / (1 + np.exp(-(-5 + 5 * age + 3 * (vibration - 0.5))))
y = (rng.random(n) < prob * 0.16).astype(int)
X = np.c_[vibration, temperature, pressure]
cut = int(n * 0.75)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(X[:cut], y[:cut])
p = model.predict_proba(X[cut:])[:, 1]
pred = (p >= 0.5).astype(int)
print("زمن التدريب والاختبار:", cut, n - cut)
print("AP آخر فترة:", round(average_precision_score(y[cut:], p), 4))
print("Recall عند 0.5:", round(recall_score(y[cut:], pred, zero_division=0), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(11)
normal = rng.normal(size=(500, 3))
later = rng.normal(size=(120, 3))
later[:8] += np.array([4.0, 4.0, -4.0])
detector = IsolationForest(contamination=0.03, random_state=42)
detector.fit(normal)
scores = -detector.score_samples(later)
order = np.argsort(scores)[::-1]
print("أعلى درجات الشذوذ:", np.round(scores[order[:8]], 3).tolist())
print("عدد النقاط المرصودة:", len(later))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
risk_score = np.array([0.08, 0.2, 0.8, 0.35, 0.61,
                       0.1, 0.42, 0.73, 0.18, 0.55])
for threshold in (0.4, 0.65):
    alert = (risk_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, alert).ravel()
    print(f"threshold={threshold}: alerts={int(alert.sum())}, "
          f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

rng = np.random.default_rng(42)
n = 1000
t = np.arange(n)
p = 1 / (1 + np.exp(-(-5 + 6 * t / n)))
y = (rng.random(n) < p * 0.15).astype(int)
for fraction in (0.7, 0.8):
    cut = int(n * fraction)
    print(fraction, "أعطال تدريب/اختبار:",
          int(y[:cut].sum()), int(y[cut:].sum()))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch104.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
reference_a = rng.normal(0, 1, (400, 2))
reference_b = rng.normal(0.5, 1.2, (400, 2))
observations = np.array([[2.5, 2.5], [0.0, 0.0], [-3.0, 2.0]])
for reference in (reference_a, reference_b):
    model = IsolationForest(contamination=0.05, random_state=42).fit(reference)
    print(np.round(-model.score_samples(observations), 3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500
t = np.arange(n)
degradation = t / n
vibration = 0.4 + 0.25 * degradation + rng.normal(0, 0.08, n)
temperature = 60 + 12 * degradation + rng.normal(0, 3, n)
pressure = 100 - 10 * degradation + rng.normal(0, 4, n)
risk = 1 / (1 + np.exp(-(-5 + 5 * degradation + 4 * (vibration - 0.5))))
failure_next_24h = (rng.random(n) < risk * 0.12).astype(int)
data = pd.DataFrame({"time": t, "vibration": vibration,
                     "temperature": temperature, "pressure": pressure,
                     "failure_next_24h": failure_next_24h})
print("شكل الجدول:", data.shape)
print("معدل الهدف:", round(float(data["failure_next_24h"].mean()), 3))
print(data.tail(3).round(2).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(17)
n = 1800
t = np.arange(n)
age = t / n
vibration = 0.35 + 0.3 * age + rng.normal(0, 0.1, n)
temperature = 55 + 18 * age + rng.normal(0, 4, n)
pressure = 105 - 13 * age + rng.normal(0, 5, n)
prob = 1 / (1 + np.exp(-(-5 + 5 * age + 3 * (vibration - 0.5))))
y = (rng.random(n) < prob * 0.16).astype(int)
X = np.c_[vibration, temperature, pressure]
cut = int(n * 0.75)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(X[:cut], y[:cut])
p = model.predict_proba(X[cut:])[:, 1]
pred = (p >= 0.5).astype(int)
print("زمن التدريب والاختبار:", cut, n - cut)
print("AP آخر فترة:", round(average_precision_score(y[cut:], p), 4))
print("Recall عند 0.5:", round(recall_score(y[cut:], pred, zero_division=0), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(11)
normal = rng.normal(size=(500, 3))
later = rng.normal(size=(120, 3))
later[:8] += np.array([4.0, 4.0, -4.0])
detector = IsolationForest(contamination=0.03, random_state=42)
detector.fit(normal)
scores = -detector.score_samples(later)
order = np.argsort(scores)[::-1]
print("أعلى درجات الشذوذ:", np.round(scores[order[:8]], 3).tolist())
print("عدد النقاط المرصودة:", len(later))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
risk_score = np.array([0.08, 0.2, 0.8, 0.35, 0.61,
                       0.1, 0.42, 0.73, 0.18, 0.55])
for threshold in (0.4, 0.65):
    alert = (risk_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, alert).ravel()
    print(f"threshold={threshold}: alerts={int(alert.sum())}, "
          f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

rng = np.random.default_rng(42)
n = 1000
t = np.arange(n)
p = 1 / (1 + np.exp(-(-5 + 6 * t / n)))
y = (rng.random(n) < p * 0.15).astype(int)
for fraction in (0.7, 0.8):
    cut = int(n * fraction)
    print(fraction, "أعطال تدريب/اختبار:",
          int(y[:cut].sum()), int(y[cut:].sum()))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-104-ch104.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
reference_a = rng.normal(0, 1, (400, 2))
reference_b = rng.normal(0.5, 1.2, (400, 2))
observations = np.array([[2.5, 2.5], [0.0, 0.0], [-3.0, 2.0]])
for reference in (reference_a, reference_b):
    model = IsolationForest(contamination=0.05, random_state=42).fit(reference)
    print(np.round(-model.score_samples(observations), 3).tolist())
