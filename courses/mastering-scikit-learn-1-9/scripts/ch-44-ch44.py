#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ExtraTrees وتضمين الأشجار العشوائية
Mastering scikit-learn 1.9 — ExtraTrees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-44-ch44 (lesson 44)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-44-ch44.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ExtraTrees وتضمين الأشجار العشوائية
Mastering scikit-learn 1.9 — ExtraTrees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-44-ch44 (lesson 44)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import time

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:2000], data.target[:2000], test_size=0.3, random_state=42
)

# تدريب RandomForest
t0 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rf = time.perf_counter() - t0
rmse_rf = root_mean_squared_error(y_te, rf.predict(X_te))

# تدريب ExtraTrees
t0 = time.perf_counter()
et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_et = time.perf_counter() - t0
rmse_et = root_mean_squared_error(y_te, et.predict(X_te))

print(f"RandomForest | RMSE الاختبار: {rmse_rf:.4f} | زمن التدريب: {t_rf * 1000:.1f} ms")
print(f"ExtraTrees   | RMSE الاختبار: {rmse_et:.4f} | زمن التدريب: {t_et * 1000:.1f} ms")
print(f"فارق السرعة لصالح ExtraTrees: {t_rf / t_et:.2f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomTreesEmbedding
import numpy as np

rng = np.random.default_rng(42)
X_toy = rng.normal(size=(5, 3))

# تدريب تضمين غير مشرف بـ 4 أشجار وعمق 2
hasher = RandomTreesEmbedding(n_estimators=4, max_depth=2, sparse_output=False, random_state=42)
X_transformed = hasher.fit_transform(X_toy)

print(f"أبعاد البيانات الأصلية: {X_toy.shape}")
print(f"أبعاد مصفوفة التضمين: {X_transformed.shape}")
print(f"عدد العناصر غير الصفرية في كل صف: {X_transformed.sum(axis=1)}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# بيانات الهلالين المتداخلين (غير خطية إطلاقاً)
X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. تصنيف لوجيستي خطي عادي (سيفشل حتماً في فصل الهلالين)
lr_raw = LogisticRegression().fit(X_tr, y_tr)

# 2. تصنيف لوجيستي مسبوق بتضمين الأشجار العشوائية
pipe_embedding = Pipeline([
    ("tree_embed", RandomTreesEmbedding(n_estimators=100, max_depth=4, sparse_output=False, random_state=42)),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe_embedding.fit(X_tr, y_tr)

print(f"دقة LogisticRegression المباشر على الهلالين: {lr_raw.score(X_te, y_te):.4f} (عجز خطي)")
print(f"دقة LogisticRegression بعد RandomTreesEmbedding: {pipe_embedding.score(X_te, y_te):.4f} (تفوق غير خطي باهر!)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomTreesEmbedding
import numpy as np

# عينتان متقاربتان جداً وعينة ثالثة بعيدة
points = np.array([
    [1.0, 1.0],
    [1.05, 0.95],
    [10.0, 10.0]
])

embedder = RandomTreesEmbedding(n_estimators=100, max_depth=3, sparse_output=False, random_state=42)
encoded = embedder.fit_transform(points)

# حساب التشابه بالضرب القياسي لمصفوفات التضمين
sim_arr = np.asarray((encoded @ encoded.T) / 100.0)

print(f"نسبة تشابه العينة 1 مع العينة 2 (المتقاربة): {sim_arr[0, 1]:.2%}")
print(f"نسبة تشابه العينة 1 مع العينة 3 (البعيدة):    {sim_arr[0, 2]:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import time

X, y = load_digits(return_X_y=True)

# RandomForest
t0 = time.perf_counter()
score_rf = cross_val_score(RandomForestClassifier(n_estimators=80, random_state=42), X, y, cv=5).mean()
t_rf = time.perf_counter() - t0

# ExtraTrees
t0 = time.perf_counter()
score_et = cross_val_score(ExtraTreesClassifier(n_estimators=80, random_state=42), X, y, cv=5).mean()
t_et = time.perf_counter() - t0

print(f"RandomForest | متوسط دقة CV: {score_rf:.4f} | زمن التنفيذ: {t_rf:.2f} s")
print(f"ExtraTrees   | متوسط دقة CV: {score_et:.4f} | زمن التنفيذ: {t_et:.2f} s")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch44.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=500, factor=0.4, noise=0.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for n in [10, 30, 80, 150]:
    pipe = make_pipeline(
        RandomTreesEmbedding(n_estimators=n, max_depth=3, sparse_output=False, random_state=42),
        RidgeClassifier()
    ).fit(X_tr, y_tr)
    print(f"أشجار التضمين: {n:3d} | دقة الاختبار للنموذج الخطي: {pipe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import time

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:2000], data.target[:2000], test_size=0.3, random_state=42
)

# تدريب RandomForest
t0 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_rf = time.perf_counter() - t0
rmse_rf = root_mean_squared_error(y_te, rf.predict(X_te))

# تدريب ExtraTrees
t0 = time.perf_counter()
et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
t_et = time.perf_counter() - t0
rmse_et = root_mean_squared_error(y_te, et.predict(X_te))

print(f"RandomForest | RMSE الاختبار: {rmse_rf:.4f} | زمن التدريب: {t_rf * 1000:.1f} ms")
print(f"ExtraTrees   | RMSE الاختبار: {rmse_et:.4f} | زمن التدريب: {t_et * 1000:.1f} ms")
print(f"فارق السرعة لصالح ExtraTrees: {t_rf / t_et:.2f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomTreesEmbedding
import numpy as np

rng = np.random.default_rng(42)
X_toy = rng.normal(size=(5, 3))

# تدريب تضمين غير مشرف بـ 4 أشجار وعمق 2
hasher = RandomTreesEmbedding(n_estimators=4, max_depth=2, sparse_output=False, random_state=42)
X_transformed = hasher.fit_transform(X_toy)

print(f"أبعاد البيانات الأصلية: {X_toy.shape}")
print(f"أبعاد مصفوفة التضمين: {X_transformed.shape}")
print(f"عدد العناصر غير الصفرية في كل صف: {X_transformed.sum(axis=1)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# بيانات الهلالين المتداخلين (غير خطية إطلاقاً)
X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. تصنيف لوجيستي خطي عادي (سيفشل حتماً في فصل الهلالين)
lr_raw = LogisticRegression().fit(X_tr, y_tr)

# 2. تصنيف لوجيستي مسبوق بتضمين الأشجار العشوائية
pipe_embedding = Pipeline([
    ("tree_embed", RandomTreesEmbedding(n_estimators=100, max_depth=4, sparse_output=False, random_state=42)),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe_embedding.fit(X_tr, y_tr)

print(f"دقة LogisticRegression المباشر على الهلالين: {lr_raw.score(X_te, y_te):.4f} (عجز خطي)")
print(f"دقة LogisticRegression بعد RandomTreesEmbedding: {pipe_embedding.score(X_te, y_te):.4f} (تفوق غير خطي باهر!)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomTreesEmbedding
import numpy as np

# عينتان متقاربتان جداً وعينة ثالثة بعيدة
points = np.array([
    [1.0, 1.0],
    [1.05, 0.95],
    [10.0, 10.0]
])

embedder = RandomTreesEmbedding(n_estimators=100, max_depth=3, sparse_output=False, random_state=42)
encoded = embedder.fit_transform(points)

# حساب التشابه بالضرب القياسي لمصفوفات التضمين
sim_arr = np.asarray((encoded @ encoded.T) / 100.0)

print(f"نسبة تشابه العينة 1 مع العينة 2 (المتقاربة): {sim_arr[0, 1]:.2%}")
print(f"نسبة تشابه العينة 1 مع العينة 3 (البعيدة):    {sim_arr[0, 2]:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import time

X, y = load_digits(return_X_y=True)

# RandomForest
t0 = time.perf_counter()
score_rf = cross_val_score(RandomForestClassifier(n_estimators=80, random_state=42), X, y, cv=5).mean()
t_rf = time.perf_counter() - t0

# ExtraTrees
t0 = time.perf_counter()
score_et = cross_val_score(ExtraTreesClassifier(n_estimators=80, random_state=42), X, y, cv=5).mean()
t_et = time.perf_counter() - t0

print(f"RandomForest | متوسط دقة CV: {score_rf:.4f} | زمن التنفيذ: {t_rf:.2f} s")
print(f"ExtraTrees   | متوسط دقة CV: {score_et:.4f} | زمن التنفيذ: {t_et:.2f} s")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-44-ch44.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=500, factor=0.4, noise=0.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for n in [10, 30, 80, 150]:
    pipe = make_pipeline(
        RandomTreesEmbedding(n_estimators=n, max_depth=3, sparse_output=False, random_state=42),
        RidgeClassifier()
    ).fit(X_tr, y_tr)
    print(f"أشجار التضمين: {n:3d} | دقة الاختبار للنموذج الخطي: {pipe.score(X_te, y_te):.4f}")
