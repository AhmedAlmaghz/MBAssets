#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الأداء والتوازي والذاكرة
Mastering scikit-learn 1.9 — Chapter 97 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-97-ch97 (lesson 97)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-97-ch97.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الأداء والتوازي والذاكرة
Mastering scikit-learn 1.9 — Chapter 97 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-97-ch97 (lesson 97)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
t0 = perf_counter()
model.fit(X, y)
fit_seconds = perf_counter() - t0
t0 = perf_counter()
pred = model.predict(X)
predict_seconds = perf_counter() - t0
print(f"fit: {fit_seconds:.4f}s | predict: {predict_seconds:.4f}s")
print("عدد التنبؤات:", pred.shape[0])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
result = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy",
                        n_jobs=2, error_score="raise")
print("درجات:", result["test_score"].round(3).tolist())
print("زمن الملاءمة:", result["fit_time"].round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
from joblib import Parallel, delayed

def summarize(values):
    return (len(values), sum(values), max(values))

chunks = [list(range(start, start + 100)) for start in (0, 100, 200, 300)]
results = Parallel(n_jobs=2)(delayed(summarize)(chunk) for chunk in chunks)
print("ملخص كل جزء:", results)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix

rng = np.random.default_rng(42)
dense = (rng.random((1000, 1000)) < 0.01).astype(np.float64)
sparse = csr_matrix(dense)
dense_bytes = dense.nbytes
sparse_bytes = sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
print("الكثافة:", round(float(np.count_nonzero(dense) / dense.size), 4))
print("حجم dense:", dense_bytes, "بايت")
print("حجم CSR:", sparse_bytes, "بايت")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
for n in (50, 100, 200):
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1)
    t = perf_counter()
    model.fit(X, y)
    print(n, "أشجار، fit:", round(perf_counter() - t, 4), "ثانية")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch97.md
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
for jobs in (1, 2):
    t = perf_counter()
    r = cross_validate(model, X, y, cv=5, n_jobs=jobs,
                       scoring="balanced_accuracy")
    print(jobs, round(perf_counter() - t, 3),
          r["test_score"].round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
t0 = perf_counter()
model.fit(X, y)
fit_seconds = perf_counter() - t0
t0 = perf_counter()
pred = model.predict(X)
predict_seconds = perf_counter() - t0
print(f"fit: {fit_seconds:.4f}s | predict: {predict_seconds:.4f}s")
print("عدد التنبؤات:", pred.shape[0])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
result = cross_validate(model, X, y, cv=5, scoring="balanced_accuracy",
                        n_jobs=2, error_score="raise")
print("درجات:", result["test_score"].round(3).tolist())
print("زمن الملاءمة:", result["fit_time"].round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from joblib import Parallel, delayed

def summarize(values):
    return (len(values), sum(values), max(values))

chunks = [list(range(start, start + 100)) for start in (0, 100, 200, 300)]
results = Parallel(n_jobs=2)(delayed(summarize)(chunk) for chunk in chunks)
print("ملخص كل جزء:", results)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix

rng = np.random.default_rng(42)
dense = (rng.random((1000, 1000)) < 0.01).astype(np.float64)
sparse = csr_matrix(dense)
dense_bytes = dense.nbytes
sparse_bytes = sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
print("الكثافة:", round(float(np.count_nonzero(dense) / dense.size), 4))
print("حجم dense:", dense_bytes, "بايت")
print("حجم CSR:", sparse_bytes, "بايت")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
for n in (50, 100, 200):
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1)
    t = perf_counter()
    model.fit(X, y)
    print(n, "أشجار، fit:", round(perf_counter() - t, 4), "ثانية")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-97-ch97.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
for jobs in (1, 2):
    t = perf_counter()
    r = cross_validate(model, X, y, cv=5, n_jobs=jobs,
                       scoring="balanced_accuracy")
    print(jobs, round(perf_counter() - t, 3),
          r["test_score"].round(3).tolist())
