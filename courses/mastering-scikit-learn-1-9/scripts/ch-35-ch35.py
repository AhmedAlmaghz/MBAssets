#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصنيف اللوجيستي: من الاحتمال إلى القرار
Mastering scikit-learn 1.9 — Chapter 35 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-35-ch35 (lesson 35)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-35-ch35.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصنيف اللوجيستي: من الاحتمال إلى القرار
Mastering scikit-learn 1.9 — Chapter 35 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-35-ch35 (lesson 35)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, random_state=42))
])

pipe_lr.fit(X, y)
lr = pipe_lr.named_steps["clf"]

print(f"أبعاد مصفوفة المعاملات coef_: {lr.coef_.shape}")
print(f"قيمة الحد الثابت intercept_: {lr.intercept_[0]:.3f}")
print(f"دقة التدريب: {pipe_lr.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# اختيار فئتين فقط وميزتين للإيضاح الهندسي
X, y = load_iris(return_X_y=True)
mask = y < 2
X_bin, y_bin = X[mask, :2], y[mask]

clf = LogisticRegression(random_state=42).fit(X_bin, y_bin)

# فحص عينة قريبة من حافة القرار
sample = X_bin[:3]
scores = clf.decision_function(sample)
probs = clf.predict_proba(sample)
preds = clf.predict(sample)

print(f"{'العينة':<8} {'decision_function':>18} {'P(y=1)':>10} {'التنبؤ':>8}")
print("-" * 48)
for i in range(3):
    print(f"#{i+1:<7} {scores[i]:>18.3f} {probs[i, 1]:>10.3f} {preds[i]:>8}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

X, y = load_breast_cancer(return_X_y=True)

# مقارنة بين lbfgs السريع و saga المرن مع تنظيم ElasticNet
for solver, l1_ratio in [("lbfgs", None), ("saga", 0.5)]:
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            solver=solver, l1_ratio=l1_ratio, max_iter=2000, random_state=42
        ))
    ]).fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"المحرك {solver:<8} | l1_ratio={str(l1_ratio):<5} | الدقة: {pipe.score(X, y):.4f} | الزمن: {elapsed:.2f} ms")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_iris(return_X_y=True)
class_names = load_iris().target_names

pipe_multi = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42))
])

pipe_multi.fit(X, y)
clf_multi = pipe_multi.named_steps["clf"]

print(f"عدد الفئات المستهدفة: {len(class_names)}")
print(f"أبعاد مصفوفة المعاملات coef_: {clf_multi.coef_.shape} (فئة x ميزة)")
print("احتمالات الانتماء للفئات للعينة الأولى:")
first_prob = pipe_multi.predict_proba(X[:1])[0]
for name, prob in zip(class_names, first_prob):
    print(f"  الفئة '{name}': {prob:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

print(f"{'قيمة C':<10} {'متوسط الدقة CV':>15} {'معيار الأوزان L2':>18}")
print("-" * 46)
for c in [0.001, 0.01, 1.0, 100.0]:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
    cv_score = cross_val_score(pipe, X, y, cv=5).mean()
    pipe.fit(X, y)
    l2_norm = np.linalg.norm(pipe.named_steps["logisticregression"].coef_)
    print(f"{c:<10.3f} {cv_score:>15.4f} {l2_norm:>18.2f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch35.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.05, solver="saga", l1_ratio=1.0, max_iter=3000, random_state=42)
)
pipe.fit(X, y)
coef = pipe.named_steps["logisticregression"].coef_[0]

zero_count = np.sum(coef == 0.0)
print(f"إجمالي الميزات: {len(coef)}")
print(f"الميزات المصفرة بالكامل بواسطة عقوبة L1 في التصنيف اللوجيستي: {zero_count}")
print(f"الميزات النشطة المعتمدة في القرار: {len(coef) - zero_count}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, random_state=42))
])

pipe_lr.fit(X, y)
lr = pipe_lr.named_steps["clf"]

print(f"أبعاد مصفوفة المعاملات coef_: {lr.coef_.shape}")
print(f"قيمة الحد الثابت intercept_: {lr.intercept_[0]:.3f}")
print(f"دقة التدريب: {pipe_lr.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# اختيار فئتين فقط وميزتين للإيضاح الهندسي
X, y = load_iris(return_X_y=True)
mask = y < 2
X_bin, y_bin = X[mask, :2], y[mask]

clf = LogisticRegression(random_state=42).fit(X_bin, y_bin)

# فحص عينة قريبة من حافة القرار
sample = X_bin[:3]
scores = clf.decision_function(sample)
probs = clf.predict_proba(sample)
preds = clf.predict(sample)

print(f"{'العينة':<8} {'decision_function':>18} {'P(y=1)':>10} {'التنبؤ':>8}")
print("-" * 48)
for i in range(3):
    print(f"#{i+1:<7} {scores[i]:>18.3f} {probs[i, 1]:>10.3f} {preds[i]:>8}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

X, y = load_breast_cancer(return_X_y=True)

# مقارنة بين lbfgs السريع و saga المرن مع تنظيم ElasticNet
for solver, l1_ratio in [("lbfgs", None), ("saga", 0.5)]:
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            solver=solver, l1_ratio=l1_ratio, max_iter=2000, random_state=42
        ))
    ]).fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"المحرك {solver:<8} | l1_ratio={str(l1_ratio):<5} | الدقة: {pipe.score(X, y):.4f} | الزمن: {elapsed:.2f} ms")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_iris(return_X_y=True)
class_names = load_iris().target_names

pipe_multi = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42))
])

pipe_multi.fit(X, y)
clf_multi = pipe_multi.named_steps["clf"]

print(f"عدد الفئات المستهدفة: {len(class_names)}")
print(f"أبعاد مصفوفة المعاملات coef_: {clf_multi.coef_.shape} (فئة x ميزة)")
print("احتمالات الانتماء للفئات للعينة الأولى:")
first_prob = pipe_multi.predict_proba(X[:1])[0]
for name, prob in zip(class_names, first_prob):
    print(f"  الفئة '{name}': {prob:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

print(f"{'قيمة C':<10} {'متوسط الدقة CV':>15} {'معيار الأوزان L2':>18}")
print("-" * 46)
for c in [0.001, 0.01, 1.0, 100.0]:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
    cv_score = cross_val_score(pipe, X, y, cv=5).mean()
    pipe.fit(X, y)
    l2_norm = np.linalg.norm(pipe.named_steps["logisticregression"].coef_)
    print(f"{c:<10.3f} {cv_score:>15.4f} {l2_norm:>18.2f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-35-ch35.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.05, solver="saga", l1_ratio=1.0, max_iter=3000, random_state=42)
)
pipe.fit(X, y)
coef = pipe.named_steps["logisticregression"].coef_[0]

zero_count = np.sum(coef == 0.0)
print(f"إجمالي الميزات: {len(coef)}")
print(f"الميزات المصفرة بالكامل بواسطة عقوبة L1 في التصنيف اللوجيستي: {zero_count}")
print(f"الميزات النشطة المعتمدة في القرار: {len(coef) - zero_count}")
