#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المصنِّفات الخطية الأخرى
Mastering scikit-learn 1.9 — Chapter 36 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-36-ch36 (lesson 36)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-36-ch36.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المصنِّفات الخطية الأخرى
Mastering scikit-learn 1.9 — Chapter 36 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-36-ch36 (lesson 36)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

pipe_perc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", Perceptron(max_iter=1000, random_state=42))
])

pipe_perc.fit(X, y)
print(f"دقة البرسبترون الكلاسيكي: {pipe_perc.score(X, y):.4f}")
print(f"عدد التكرارات حتى التوقف: {pipe_perc.named_steps['clf'].n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

rng = np.random.default_rng(42)
n_features = 20

pac = PassiveAggressiveClassifier(C=1.0, random_state=42)
classes = np.array([0, 1])

# محاكاة تدفق بيانات حي عبر 5 دفعات متتابعة
for stream_batch in range(5):
    X_batch = rng.normal(size=(50, n_features))
    y_batch = (X_batch[:, 0] + X_batch[:, 1] > 0).astype(int)

    # تحديث لحظي تراكمي
    pac.partial_fit(X_batch, y_batch, classes=classes)
    acc = pac.score(X_batch, y_batch)
    print(f"الدفعة التدفقية #{stream_batch+1}: دقة التكيف اللحظي = {acc:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# توليد بيانات ضخمة (10,000 عينة)
X, y = make_classification(n_samples=10000, n_features=50, random_state=42)

for loss in ["hinge", "log_loss", "modified_huber"]:
    pipe_sgd = Pipeline([
        ("sc", StandardScaler()),
        ("clf", SGDClassifier(loss=loss, alpha=1e-4, max_iter=1000, random_state=42))
    ])
    pipe_sgd.fit(X, y)
    print(f"خسارة {loss:<15} | دقة التدريب: {pipe_sgd.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

X, y = make_classification(n_samples=2000, n_features=100, n_classes=4,
                           n_informative=40, random_state=42)
X_sc = StandardScaler().fit_transform(X)

# مقارنة زمن التدريب بين RidgeClassifier و LogisticRegression
t0 = time.perf_counter()
ridge_clf = RidgeClassifier(alpha=1.0).fit(X_sc, y)
t_ridge = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
lr_clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_sc, y)
t_lr = (time.perf_counter() - t0) * 1000

print(f"RidgeClassifier   | الدقة: {ridge_clf.score(X_sc, y):.4f} | الزمن: {t_ridge:.2f} ms")
print(f"LogisticRegression | الدقة: {lr_clf.score(X_sc, y):.4f} | الزمن: {t_lr:.2f} ms")
print(f"RidgeClassifier أسرع بنحو {t_lr / t_ridge:.1f} مرة!")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
sgd = SGDClassifier(loss="log_loss", random_state=42)
classes = np.unique(y)

# تجزئة البيانات إلى 10 دفعات
batch_size = 200
accuracies = []
for i in range(10):
    start, end = i * batch_size, (i + 1) * batch_size
    sgd.partial_fit(X[start:end], y[start:end], classes=classes)
    # تقييم على الدفعة التالية
    if end < len(X):
        acc = sgd.score(X[end:end+batch_size], y[end:end+batch_size])
        accuracies.append(acc)

print(f"دقة الدفعة الأولى: {accuracies[0]:.2%}")
print(f"دقة الدفعة الأخيرة: {accuracies[-1]:.2%}")
print(f"التحسن مع توالي التدفق: {accuracies[-1] - accuracies[0]:+.2%}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part04/ch36.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
ridge = RidgeClassifier()

# RidgeClassifier لا يمتلك predict_proba لأنه انحدار مربعات صغرى
print(f"هل يمتلك RidgeClassifier دالة predict_proba؟ {hasattr(ridge, 'predict_proba')}")

# الحل: تغليفه بمقدر المعايرة الاحتمالية CalibratedClassifierCV
calibrated_ridge = CalibratedClassifierCV(estimator=ridge, cv=3).fit(X, y)
print(f"هل يمتلك النموذج المعاير predict_proba؟ {hasattr(calibrated_ridge, 'predict_proba')}")
print("احتمالات أول عينة:", np.round(calibrated_ridge.predict_proba(X[:1])[0], 3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

pipe_perc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", Perceptron(max_iter=1000, random_state=42))
])

pipe_perc.fit(X, y)
print(f"دقة البرسبترون الكلاسيكي: {pipe_perc.score(X, y):.4f}")
print(f"عدد التكرارات حتى التوقف: {pipe_perc.named_steps['clf'].n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

rng = np.random.default_rng(42)
n_features = 20

pac = PassiveAggressiveClassifier(C=1.0, random_state=42)
classes = np.array([0, 1])

# محاكاة تدفق بيانات حي عبر 5 دفعات متتابعة
for stream_batch in range(5):
    X_batch = rng.normal(size=(50, n_features))
    y_batch = (X_batch[:, 0] + X_batch[:, 1] > 0).astype(int)

    # تحديث لحظي تراكمي
    pac.partial_fit(X_batch, y_batch, classes=classes)
    acc = pac.score(X_batch, y_batch)
    print(f"الدفعة التدفقية #{stream_batch+1}: دقة التكيف اللحظي = {acc:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# توليد بيانات ضخمة (10,000 عينة)
X, y = make_classification(n_samples=10000, n_features=50, random_state=42)

for loss in ["hinge", "log_loss", "modified_huber"]:
    pipe_sgd = Pipeline([
        ("sc", StandardScaler()),
        ("clf", SGDClassifier(loss=loss, alpha=1e-4, max_iter=1000, random_state=42))
    ])
    pipe_sgd.fit(X, y)
    print(f"خسارة {loss:<15} | دقة التدريب: {pipe_sgd.score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

X, y = make_classification(n_samples=2000, n_features=100, n_classes=4,
                           n_informative=40, random_state=42)
X_sc = StandardScaler().fit_transform(X)

# مقارنة زمن التدريب بين RidgeClassifier و LogisticRegression
t0 = time.perf_counter()
ridge_clf = RidgeClassifier(alpha=1.0).fit(X_sc, y)
t_ridge = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
lr_clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_sc, y)
t_lr = (time.perf_counter() - t0) * 1000

print(f"RidgeClassifier   | الدقة: {ridge_clf.score(X_sc, y):.4f} | الزمن: {t_ridge:.2f} ms")
print(f"LogisticRegression | الدقة: {lr_clf.score(X_sc, y):.4f} | الزمن: {t_lr:.2f} ms")
print(f"RidgeClassifier أسرع بنحو {t_lr / t_ridge:.1f} مرة!")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
sgd = SGDClassifier(loss="log_loss", random_state=42)
classes = np.unique(y)

# تجزئة البيانات إلى 10 دفعات
batch_size = 200
accuracies = []
for i in range(10):
    start, end = i * batch_size, (i + 1) * batch_size
    sgd.partial_fit(X[start:end], y[start:end], classes=classes)
    # تقييم على الدفعة التالية
    if end < len(X):
        acc = sgd.score(X[end:end+batch_size], y[end:end+batch_size])
        accuracies.append(acc)

print(f"دقة الدفعة الأولى: {accuracies[0]:.2%}")
print(f"دقة الدفعة الأخيرة: {accuracies[-1]:.2%}")
print(f"التحسن مع توالي التدفق: {accuracies[-1] - accuracies[0]:+.2%}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-36-ch36.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
ridge = RidgeClassifier()

# RidgeClassifier لا يمتلك predict_proba لأنه انحدار مربعات صغرى
print(f"هل يمتلك RidgeClassifier دالة predict_proba؟ {hasattr(ridge, 'predict_proba')}")

# الحل: تغليفه بمقدر المعايرة الاحتمالية CalibratedClassifierCV
calibrated_ridge = CalibratedClassifierCV(estimator=ridge, cv=3).fit(X, y)
print(f"هل يمتلك النموذج المعاير predict_proba؟ {hasattr(calibrated_ridge, 'predict_proba')}")
print("احتمالات أول عينة:", np.round(calibrated_ridge.predict_proba(X[:1])[0], 3))
