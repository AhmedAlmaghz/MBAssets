#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تعزيز التدرّج: شرح كلمة بكلمة
Mastering scikit-learn 1.9 — Chapter 46 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-46-ch46 (lesson 46)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-46-ch46.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تعزيز التدرّج: شرح كلمة بكلمة
Mastering scikit-learn 1.9 — Chapter 46 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-46-ch46 (lesson 46)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(240, 4))
y = X[:, 0] * np.sin(np.pi * X[:, 1]) + 0.3 * rng.normal(size=240)
nu = 0.5

# البداية: أفضل تقدير ثابت = متوسط y
F = np.full(len(y), y.mean())
for m in range(1, 5):
    r = y - F                                  # البواقي: المشتق السالب للخسارة المربعة
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, r)
    F = F + nu * tree.predict(X)               # الخطوة الجمعية الصغيرة
    print(f"بعد الشجرة {m}: RMSE التدريب = {root_mean_squared_error(y, F):.4f}")

gb = GradientBoostingRegressor(
    n_estimators=4, learning_rate=0.5, max_depth=2, random_state=42,
).fit(X, y)
manual = float(np.max(np.abs(gb.predict(X) - F)))
print(f"أكبر فرق بين البناء اليدوي و scikit-learn: {manual:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

gb = GradientBoostingClassifier(
    n_estimators=500, n_iter_no_change=10, validation_fraction=0.15, random_state=42,
).fit(X_tr, y_tr)
print(f"طلبت 500 شجرة فبنى فعلياً {gb.n_estimators_} فقط بفضل التوقف المبكر")
print(f"دقة الاختبار: {gb.score(X_te, y_te):.4f}")

# مسار التعلم: دقة الاختبار بعد كل شجرة عبر staged_predict
gb100 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=42).fit(X_tr, y_tr)
staged = [accuracy_score(y_te, p) for p in gb100.staged_predict(X_te)]
for k in [1, 5, 10, 25, 50, 100]:
    print(f"بعد {k:3d} شجرة | دقة الاختبار: {staged[k-1]:.4f}")
print(f"أفضل نقطة: {max(staged):.4f} عند الشجرة {int(np.argmax(staged)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

X, y = make_regression(n_samples=800, n_features=8, n_informative=5, noise=12, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
y_tr = y_tr.copy()
y_tr[:40] += 120.0                       # 5% شواذب في التدريب فقط

for loss in ["squared_error", "absolute_error"]:
    reg = GradientBoostingRegressor(loss=loss, n_estimators=120, random_state=42).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, reg.predict(X_te))
    mae = mean_absolute_error(y_te, reg.predict(X_te))
    print(f"loss={loss:15s} | RMSE نظيف: {rmse:7.2f} | MAE نظيف: {mae:7.2f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=15, n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for ss in [1.0, 0.7, 0.4]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(n_estimators=150, subsample=ss, random_state=42).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"subsample={ss} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for lr, n in [(1.0, 50), (0.5, 100), (0.1, 300), (0.05, 600)]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n, max_depth=2, random_state=42,
    ).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"lr={lr:4} | n={n:3d} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

X, y = make_friedman1(n_samples=1000, n_features=10, noise=5.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gb = GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X_tr, y_tr)
staged = [root_mean_squared_error(y_te, p) for p in gb.staged_predict(X_te)]
best_m = int(np.argmin(staged)) + 1
print(f"أدنى RMSE اختبار: {staged[best_m-1]:.4f} عند {best_m} شجرة")
print(f"RMSE النموذج الكامل (300 شجرة): {staged[-1]:.4f}")
print(f"الوفر: {300 - best_m} شجرة يمكن حذفها بلا كلفة أداء")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch46.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for lr, n in [(1.0, 50), (0.5, 100), (0.1, 300), (0.05, 600)]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n, max_depth=2,
        subsample=0.7, random_state=42,
    ).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"lr={lr:4} | n={n:3d} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(240, 4))
y = X[:, 0] * np.sin(np.pi * X[:, 1]) + 0.3 * rng.normal(size=240)
nu = 0.5

# البداية: أفضل تقدير ثابت = متوسط y
F = np.full(len(y), y.mean())
for m in range(1, 5):
    r = y - F                                  # البواقي: المشتق السالب للخسارة المربعة
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, r)
    F = F + nu * tree.predict(X)               # الخطوة الجمعية الصغيرة
    print(f"بعد الشجرة {m}: RMSE التدريب = {root_mean_squared_error(y, F):.4f}")

gb = GradientBoostingRegressor(
    n_estimators=4, learning_rate=0.5, max_depth=2, random_state=42,
).fit(X, y)
manual = float(np.max(np.abs(gb.predict(X) - F)))
print(f"أكبر فرق بين البناء اليدوي و scikit-learn: {manual:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

gb = GradientBoostingClassifier(
    n_estimators=500, n_iter_no_change=10, validation_fraction=0.15, random_state=42,
).fit(X_tr, y_tr)
print(f"طلبت 500 شجرة فبنى فعلياً {gb.n_estimators_} فقط بفضل التوقف المبكر")
print(f"دقة الاختبار: {gb.score(X_te, y_te):.4f}")

# مسار التعلم: دقة الاختبار بعد كل شجرة عبر staged_predict
gb100 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=42).fit(X_tr, y_tr)
staged = [accuracy_score(y_te, p) for p in gb100.staged_predict(X_te)]
for k in [1, 5, 10, 25, 50, 100]:
    print(f"بعد {k:3d} شجرة | دقة الاختبار: {staged[k-1]:.4f}")
print(f"أفضل نقطة: {max(staged):.4f} عند الشجرة {int(np.argmax(staged)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

X, y = make_regression(n_samples=800, n_features=8, n_informative=5, noise=12, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
y_tr = y_tr.copy()
y_tr[:40] += 120.0                       # 5% شواذب في التدريب فقط

for loss in ["squared_error", "absolute_error"]:
    reg = GradientBoostingRegressor(loss=loss, n_estimators=120, random_state=42).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, reg.predict(X_te))
    mae = mean_absolute_error(y_te, reg.predict(X_te))
    print(f"loss={loss:15s} | RMSE نظيف: {rmse:7.2f} | MAE نظيف: {mae:7.2f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=15, n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for ss in [1.0, 0.7, 0.4]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(n_estimators=150, subsample=ss, random_state=42).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"subsample={ss} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for lr, n in [(1.0, 50), (0.5, 100), (0.1, 300), (0.05, 600)]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n, max_depth=2, random_state=42,
    ).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"lr={lr:4} | n={n:3d} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

X, y = make_friedman1(n_samples=1000, n_features=10, noise=5.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gb = GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X_tr, y_tr)
staged = [root_mean_squared_error(y_te, p) for p in gb.staged_predict(X_te)]
best_m = int(np.argmin(staged)) + 1
print(f"أدنى RMSE اختبار: {staged[best_m-1]:.4f} عند {best_m} شجرة")
print(f"RMSE النموذج الكامل (300 شجرة): {staged[-1]:.4f}")
print(f"الوفر: {300 - best_m} شجرة يمكن حذفها بلا كلفة أداء")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-46-ch46.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for lr, n in [(1.0, 50), (0.5, 100), (0.1, 300), (0.05, 600)]:
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n, max_depth=2,
        subsample=0.7, random_state=42,
    ).fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    print(f"lr={lr:4} | n={n:3d} | الزمن: {t*1000:6.0f} ms |"
          f" تدريب: {gb.score(X_tr, y_tr):.4f} | اختبار: {gb.score(X_te, y_te):.4f}")
