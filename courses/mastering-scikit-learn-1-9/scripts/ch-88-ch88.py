#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التخزين المؤقت والخطوط الثقيلة
Mastering scikit-learn 1.9 — Chapter 88 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-88-ch88 (lesson 88)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-88-ch88.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التخزين المؤقت والخطوط الثقيلة
Mastering scikit-learn 1.9 — Chapter 88 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-88-ch88 (lesson 88)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch88.md
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as cache_dir:
    memory = Memory(location=cache_dir, verbose=0)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("reduce", PCA(n_components=8, random_state=42)),
        ("clf", LogisticRegression(max_iter=3000)),
    ], memory=memory)
    pipe.fit(X, y)
    print("شكل التمثيل:", pipe.named_steps["reduce"].transform(
        pipe.named_steps["scale"].transform(X)).shape)
    print("الخطوات:", list(pipe.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch88.md
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from time import perf_counter
from joblib import Memory
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=600, n_features=30,
                           n_informative=12, random_state=42)
with TemporaryDirectory() as cache_dir:
    cached = Pipeline([("scale", StandardScaler()),
                       ("pca", PCA(n_components=10, random_state=42)),
                       ("clf", LogisticRegression(max_iter=1000))],
                      memory=Memory(cache_dir, verbose=0))
    t0 = perf_counter()
    cached.fit(X, y)
    first = perf_counter() - t0
    t0 = perf_counter()
    cached.fit(X, y)
    second = perf_counter() - t0
    print(f"زمن الملاءمة الأولى: {first:.4f} ثانية")
    print(f"زمن الملاءمة الثانية: {second:.4f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch88.md
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as cache_dir:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", LogisticRegression(max_iter=3000)),
    ], memory=Memory(cache_dir, verbose=0))
    grid = GridSearchCV(
        pipe, {"pca__n_components": [5, 10], "clf__C": [0.1, 1.0]},
        cv=3, scoring="balanced_accuracy", n_jobs=1)
    grid.fit(X, y)
    print("أفضل الإعدادات:", grid.best_params_)
    print("أفضل متوسط تحقق:", round(grid.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch88.md
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as d:
    p = Pipeline([("scale", StandardScaler()),
                  ("pca", PCA(n_components=5, random_state=42)),
                  ("clf", LogisticRegression(max_iter=3000))],
                 memory=Memory(d, verbose=0))
    p.fit(X, y)
    p.fit(X, y)
    print("الخطوات الملائمة:", list(p.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch88.md
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from time import perf_counter
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
def build(memory):
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=8, random_state=42)),
                     ("clf", LogisticRegression(max_iter=3000))],
                    memory=memory)
with TemporaryDirectory() as d:
    for memory in (None, Memory(d, verbose=0)):
        model = build(memory)
        t = perf_counter()
        model.fit(X, y)
        print("memory:", memory is not None, "seconds:",
              round(perf_counter() - t, 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-88-ch88.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as cache_dir:
    memory = Memory(location=cache_dir, verbose=0)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("reduce", PCA(n_components=8, random_state=42)),
        ("clf", LogisticRegression(max_iter=3000)),
    ], memory=memory)
    pipe.fit(X, y)
    print("شكل التمثيل:", pipe.named_steps["reduce"].transform(
        pipe.named_steps["scale"].transform(X)).shape)
    print("الخطوات:", list(pipe.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-88-ch88.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from time import perf_counter
from joblib import Memory
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=600, n_features=30,
                           n_informative=12, random_state=42)
with TemporaryDirectory() as cache_dir:
    cached = Pipeline([("scale", StandardScaler()),
                       ("pca", PCA(n_components=10, random_state=42)),
                       ("clf", LogisticRegression(max_iter=1000))],
                      memory=Memory(cache_dir, verbose=0))
    t0 = perf_counter()
    cached.fit(X, y)
    first = perf_counter() - t0
    t0 = perf_counter()
    cached.fit(X, y)
    second = perf_counter() - t0
    print(f"زمن الملاءمة الأولى: {first:.4f} ثانية")
    print(f"زمن الملاءمة الثانية: {second:.4f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-88-ch88.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as cache_dir:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", LogisticRegression(max_iter=3000)),
    ], memory=Memory(cache_dir, verbose=0))
    grid = GridSearchCV(
        pipe, {"pca__n_components": [5, 10], "clf__C": [0.1, 1.0]},
        cv=3, scoring="balanced_accuracy", n_jobs=1)
    grid.fit(X, y)
    print("أفضل الإعدادات:", grid.best_params_)
    print("أفضل متوسط تحقق:", round(grid.best_score_, 4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-88-ch88.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
with TemporaryDirectory() as d:
    p = Pipeline([("scale", StandardScaler()),
                  ("pca", PCA(n_components=5, random_state=42)),
                  ("clf", LogisticRegression(max_iter=3000))],
                 memory=Memory(d, verbose=0))
    p.fit(X, y)
    p.fit(X, y)
    print("الخطوات الملائمة:", list(p.named_steps))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-88-ch88.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from tempfile import TemporaryDirectory
from time import perf_counter
from joblib import Memory
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
def build(memory):
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=8, random_state=42)),
                     ("clf", LogisticRegression(max_iter=3000))],
                    memory=memory)
with TemporaryDirectory() as d:
    for memory in (None, Memory(d, verbose=0)):
        model = build(memory)
        t = perf_counter()
        model.fit(X, y)
        print("memory:", memory is not None, "seconds:",
              round(perf_counter() - t, 4))
