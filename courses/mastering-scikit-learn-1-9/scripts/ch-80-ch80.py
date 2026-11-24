#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منحنى التعلّم ومنحنى التحقق
Mastering scikit-learn 1.9 — Chapter 80 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-80-ch80 (lesson 80)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-80-ch80.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منحنى التعلّم ومنحنى التحقق
Mastering scikit-learn 1.9 — Chapter 80 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-80-ch80 (lesson 80)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=800, n_features=14, n_informative=7,
                           n_redundant=3, class_sep=0.55, random_state=42)
sizes = np.linspace(0.1, 1.0, 6)

models = [("شجرة عميقة", DecisionTreeClassifier(random_state=42)),
          ("لوجستي", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
          ("غابة", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]

for name, est in models:
    tr, trs, tes = learning_curve(est, X, y, train_sizes=sizes, cv=5,
                                  scoring="accuracy")
    a, b = trs.mean(axis=1), tes.mean(axis=1)
    print(f"{name}: أول نقطة train={a[0]:.3f} val={b[0]:.3f} | "
          f"آخر نقطة train={a[-1]:.3f} val={b[-1]:.3f} | فجوة نهائية={a[-1]-b[-1]:.3f} | "
          f"قمة التحقق={b.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
depths = np.arange(1, 16)

trs, tes = validation_curve(DecisionTreeClassifier(random_state=42), X, y,
                            param_name="max_depth", param_range=depths,
                            cv=5, scoring="accuracy")
a, b = trs.mean(axis=1), tes.mean(axis=1)
for d in [1, 2, 6, 10, 15]:
    i = d - 1
    print(f"depth={d:2d}: train={a[i]:.4f} val={b[i]:.4f}")
best = depths[np.argmax(b)]
print(f"النافذة الذهبية: depth={best} بدرجة تحقق {b.max():.4f}")
print(f"الفجوة عند القمة: {a[best-1]-b[best-1]:.4f} مقابل {a[-1]-b[-1]:.4f} عند العمق 15")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2400, n_features=12, n_informative=6,
                           n_redundant=3, class_sep=0.7, random_state=42)
sizes = np.array([150, 400, 800, 1200, 1500])

tr, trs, tes = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
    train_sizes=sizes, cv=5, scoring="accuracy", shuffle=True, random_state=42)
prev = None
for n, v in zip(tr, tes.mean(axis=1)):
    delta = "" if prev is None else f" (فارق {v - prev:+.4f})"
    print(f"n={n:5d} | درجة التحقق={v:.4f}{delta}")
    prev = v

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2400, n_features=12, n_informative=6,
                           random_state=42)
tr, trs, tes, fit_t, score_t = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
    train_sizes=np.array([300, 800, 1500]), cv=5, scoring="accuracy",
    return_times=True, shuffle=True, random_state=42)
for n, t in zip(tr, fit_t.mean(axis=1)):
    print(f"n={n:5d} | متوسط زمن التدريب={t * 1000:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
sizes = np.linspace(0.1, 1.0, 6)

for name, est in [("شجرة ضحلة (قبل العلاج)", DecisionTreeClassifier(max_depth=2, random_state=42)),
                  ("SVC نواة RBF (بعد العلاج)", make_pipeline(StandardScaler(), SVC()))]:
    tr, trs, tes = learning_curve(est, X, y, train_sizes=sizes, cv=5,
                                  scoring="accuracy")
    a, b = trs.mean(axis=1), tes.mean(axis=1)
    print(f"{name}: train النهائي={a[-1]:.4f} | val النهائي={b[-1]:.4f} | فجوة={a[-1]-b[-1]:.4f}")
print("التشخيص: فجوة تافهة وقمة تعثرت مبكراً = انحياز يحده النموذج؛ والعلاج أغنى لا بيانات")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch80.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
k_values = np.arange(1, 16)

trs, tes = validation_curve(
    make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y,
    param_name="kneighborsclassifier__n_neighbors", param_range=k_values,
    cv=5, scoring="accuracy")
a, b = trs.mean(axis=1), tes.mean(axis=1)
for k in [1, 3, 9, 15]:
    print(f"k={k:2d}: train={a[k-1]:.4f} val={b[k-1]:.4f}")
best = k_values[np.argmax(b)]
print(f"النافذة الذهبية: k={best} بدرجة تحقق {b.max():.4f}")
print(f"عند k=1: فجوة كاملة (train={a[0]:.4f}) — النموذج يحفظ ويفشل في التعميم")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=800, n_features=14, n_informative=7,
                           n_redundant=3, class_sep=0.55, random_state=42)
sizes = np.linspace(0.1, 1.0, 6)

models = [("شجرة عميقة", DecisionTreeClassifier(random_state=42)),
          ("لوجستي", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
          ("غابة", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]

for name, est in models:
    tr, trs, tes = learning_curve(est, X, y, train_sizes=sizes, cv=5,
                                  scoring="accuracy")
    a, b = trs.mean(axis=1), tes.mean(axis=1)
    print(f"{name}: أول نقطة train={a[0]:.3f} val={b[0]:.3f} | "
          f"آخر نقطة train={a[-1]:.3f} val={b[-1]:.3f} | فجوة نهائية={a[-1]-b[-1]:.3f} | "
          f"قمة التحقق={b.max():.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
depths = np.arange(1, 16)

trs, tes = validation_curve(DecisionTreeClassifier(random_state=42), X, y,
                            param_name="max_depth", param_range=depths,
                            cv=5, scoring="accuracy")
a, b = trs.mean(axis=1), tes.mean(axis=1)
for d in [1, 2, 6, 10, 15]:
    i = d - 1
    print(f"depth={d:2d}: train={a[i]:.4f} val={b[i]:.4f}")
best = depths[np.argmax(b)]
print(f"النافذة الذهبية: depth={best} بدرجة تحقق {b.max():.4f}")
print(f"الفجوة عند القمة: {a[best-1]-b[best-1]:.4f} مقابل {a[-1]-b[-1]:.4f} عند العمق 15")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2400, n_features=12, n_informative=6,
                           n_redundant=3, class_sep=0.7, random_state=42)
sizes = np.array([150, 400, 800, 1200, 1500])

tr, trs, tes = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
    train_sizes=sizes, cv=5, scoring="accuracy", shuffle=True, random_state=42)
prev = None
for n, v in zip(tr, tes.mean(axis=1)):
    delta = "" if prev is None else f" (فارق {v - prev:+.4f})"
    print(f"n={n:5d} | درجة التحقق={v:.4f}{delta}")
    prev = v

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2400, n_features=12, n_informative=6,
                           random_state=42)
tr, trs, tes, fit_t, score_t = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
    train_sizes=np.array([300, 800, 1500]), cv=5, scoring="accuracy",
    return_times=True, shuffle=True, random_state=42)
for n, t in zip(tr, fit_t.mean(axis=1)):
    print(f"n={n:5d} | متوسط زمن التدريب={t * 1000:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
sizes = np.linspace(0.1, 1.0, 6)

for name, est in [("شجرة ضحلة (قبل العلاج)", DecisionTreeClassifier(max_depth=2, random_state=42)),
                  ("SVC نواة RBF (بعد العلاج)", make_pipeline(StandardScaler(), SVC()))]:
    tr, trs, tes = learning_curve(est, X, y, train_sizes=sizes, cv=5,
                                  scoring="accuracy")
    a, b = trs.mean(axis=1), tes.mean(axis=1)
    print(f"{name}: train النهائي={a[-1]:.4f} | val النهائي={b[-1]:.4f} | فجوة={a[-1]-b[-1]:.4f}")
print("التشخيص: فجوة تافهة وقمة تعثرت مبكراً = انحياز يحده النموذج؛ والعلاج أغنى لا بيانات")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-80-ch80.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
k_values = np.arange(1, 16)

trs, tes = validation_curve(
    make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y,
    param_name="kneighborsclassifier__n_neighbors", param_range=k_values,
    cv=5, scoring="accuracy")
a, b = trs.mean(axis=1), tes.mean(axis=1)
for k in [1, 3, 9, 15]:
    print(f"k={k:2d}: train={a[k-1]:.4f} val={b[k-1]:.4f}")
best = k_values[np.argmax(b)]
print(f"النافذة الذهبية: k={best} بدرجة تحقق {b.max():.4f}")
print(f"عند k=1: فجوة كاملة (train={a[0]:.4f}) — النموذج يحفظ ويفشل في التعميم")
