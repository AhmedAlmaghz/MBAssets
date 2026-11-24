#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — SVR و OneClassSVM
Mastering scikit-learn 1.9 — SVR

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-53-ch53 (lesson 53)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-53-ch53.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — SVR و OneClassSVM
Mastering scikit-learn 1.9 — SVR

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-53-ch53 (lesson 53)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'epsilon':>8} | {'متجهات داعمة':>12} | {'MSE':>8} | {'R2':>7}")
print("-" * 46)
for eps in [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]:
    svr = make_pipeline(StandardScaler(),
                        SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=eps))
    svr.fit(X_tr, y_tr)
    pred = svr.predict(X_te)
    print(f"{eps:>8.2f} | {len(svr[-1].support_):>12} | "
          f"{mean_squared_error(y_te, pred):>8.3f} | {r2_score(y_te, pred):>7.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=600, n_features=8, noise=2.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

lin = LinearRegression().fit(X_tr, y_tr)
svr = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=1.0)).fit(X_tr, y_tr)

print(f"LinearRegression | R2 التدريب: {lin.score(X_tr, y_tr):.4f} | "
      f"R2 الاختبار: {lin.score(X_te, y_te):.4f}")
print(f"SVR(kernel='rbf') | R2 التدريب: {svr.score(X_tr, y_tr):.4f} | "
      f"R2 الاختبار: {svr.score(X_te, y_te):.4f}")
print(f"متجهات داعمة: {len(svr[-1].support_)} من أصل {len(X_tr)}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = (np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)) * 1000.0 + 50000.0

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

raw = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, gamma=0.1)).fit(X_tr, y_tr)
wrapped = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(
        regressor=SVR(kernel="rbf", C=10.0, gamma=0.1),
        transformer=StandardScaler(),
    ),
).fit(X_tr, y_tr)

print(f"هدف خام (بآلاف الريالات)      | R2: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw[-1].support_)}")
print(f"مع TransformedTargetRegressor | R2: {wrapped.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(wrapped[-1].regressor_.support_)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(42)
clean = rng.normal([0.0, 0.0], 1.0, (300, 2))            # تدريب: نمط سليم فقط

oc = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(clean)

new = rng.normal([0.0, 0.0], 1.0, (200, 2))              # وافدون سليمون
intruders = rng.uniform(-4.5, 4.5, (50, 2))              # متسللون خارج النمط
eval_X = np.vstack([new, intruders])
pred = oc.predict(eval_X)

accept = (pred[:200] == 1).mean()
caught = (pred[200:] == -1).mean()
rejected_train = (oc.predict(clean) == -1).mean()
print(f"قبول الوافدين السليمين: {accept:.1%}")
print(f"اصطياد المتسللين: {caught:.1%}")
print(f"نسبة رفض بيانات التدريب نفسها (المتوقع ≈ nu = 0.05): {rejected_train:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(42)
clean = rng.normal([0.0, 0.0], 1.0, (300, 2))
noise = rng.uniform(-4.0, 4.0, (30, 2))                  # 9% شواذب تسربت للتدريب
contaminated = np.vstack([clean, noise])

novelty = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(clean)
outlier = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(contaminated)

new = rng.normal([0.0, 0.0], 1.0, (200, 2))
intruders = rng.uniform(-4.0, 4.0, (50, 2))
eval_X = np.vstack([new, intruders])

for name, model in [("fit على نظيف  (novelty)", novelty),
                    ("fit على ملوث (outlier)", outlier)]:
    pred = model.predict(eval_X)
    print(f"{name}: قبول السليمين {((pred[:200] == 1).mean()):.1%} | "
          f"اصطياد الشواذب {((pred[200:] == -1).mean()):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

models = [
    ("SVR", make_pipeline(StandardScaler(),
                          SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=0.1))),
    ("KernelRidge", make_pipeline(StandardScaler(),
                                  KernelRidge(kernel="rbf", gamma=0.1, alpha=0.01))),
]
for name, model in models:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    mse = mean_squared_error(y_te, model.predict(X_te))
    if name == "SVR":
        n_vec = len(model[-1].support_)
    else:
        n_vec = int(model[-1].dual_coef_.size)           # كل عينة تحمل معاملاً
    print(f"{name:>12} | زمن التدريب: {dt * 1000:7.1f} ms | "
          f"MSE: {mse:.4f} | معاملات الحل: {n_vec}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = fetch_california_housing()
X, y = data.data[:1200], data.target[:1200]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'epsilon':>8} | {'متجهات داعمة':>12} | {'R2 الاختبار':>12}")
print("-" * 40)
for eps in [0.01, 0.05, 0.1, 0.3, 0.5]:
    svr = make_pipeline(StandardScaler(),
                        SVR(kernel="rbf", C=10.0, epsilon=eps)).fit(X_tr, y_tr)
    print(f"{eps:>8.2f} | {len(svr[-1].support_):>12} | "
          f"{svr.score(X_te, y_te):>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch53.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

X, y = load_digits(return_X_y=True)
zeros = X[y == 0]
rest = X[y != 0]

print(f"{'gamma':>7} | رفض الرقم 0 | رفض الأرقام الأخرى")
print("-" * 44)
for g in [0.05, 0.01, 0.002]:
    oc = make_pipeline(StandardScaler(),
                       OneClassSVM(kernel="rbf", nu=0.05, gamma=g)).fit(zeros)
    rej0 = (oc.predict(zeros) == -1).mean()
    rejrest = (oc.predict(rest) == -1).mean()
    print(f"{g:>7.3f} | {rej0:>10.1%} | {rejrest:>13.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'epsilon':>8} | {'متجهات داعمة':>12} | {'MSE':>8} | {'R2':>7}")
print("-" * 46)
for eps in [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]:
    svr = make_pipeline(StandardScaler(),
                        SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=eps))
    svr.fit(X_tr, y_tr)
    pred = svr.predict(X_te)
    print(f"{eps:>8.2f} | {len(svr[-1].support_):>12} | "
          f"{mean_squared_error(y_te, pred):>8.3f} | {r2_score(y_te, pred):>7.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=600, n_features=8, noise=2.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

lin = LinearRegression().fit(X_tr, y_tr)
svr = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=1.0)).fit(X_tr, y_tr)

print(f"LinearRegression | R2 التدريب: {lin.score(X_tr, y_tr):.4f} | "
      f"R2 الاختبار: {lin.score(X_te, y_te):.4f}")
print(f"SVR(kernel='rbf') | R2 التدريب: {svr.score(X_tr, y_tr):.4f} | "
      f"R2 الاختبار: {svr.score(X_te, y_te):.4f}")
print(f"متجهات داعمة: {len(svr[-1].support_)} من أصل {len(X_tr)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = (np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)) * 1000.0 + 50000.0

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

raw = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, gamma=0.1)).fit(X_tr, y_tr)
wrapped = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(
        regressor=SVR(kernel="rbf", C=10.0, gamma=0.1),
        transformer=StandardScaler(),
    ),
).fit(X_tr, y_tr)

print(f"هدف خام (بآلاف الريالات)      | R2: {raw.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(raw[-1].support_)}")
print(f"مع TransformedTargetRegressor | R2: {wrapped.score(X_te, y_te):.4f} | "
      f"متجهات داعمة: {len(wrapped[-1].regressor_.support_)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(42)
clean = rng.normal([0.0, 0.0], 1.0, (300, 2))            # تدريب: نمط سليم فقط

oc = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(clean)

new = rng.normal([0.0, 0.0], 1.0, (200, 2))              # وافدون سليمون
intruders = rng.uniform(-4.5, 4.5, (50, 2))              # متسللون خارج النمط
eval_X = np.vstack([new, intruders])
pred = oc.predict(eval_X)

accept = (pred[:200] == 1).mean()
caught = (pred[200:] == -1).mean()
rejected_train = (oc.predict(clean) == -1).mean()
print(f"قبول الوافدين السليمين: {accept:.1%}")
print(f"اصطياد المتسللين: {caught:.1%}")
print(f"نسبة رفض بيانات التدريب نفسها (المتوقع ≈ nu = 0.05): {rejected_train:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(42)
clean = rng.normal([0.0, 0.0], 1.0, (300, 2))
noise = rng.uniform(-4.0, 4.0, (30, 2))                  # 9% شواذب تسربت للتدريب
contaminated = np.vstack([clean, noise])

novelty = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(clean)
outlier = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.5).fit(contaminated)

new = rng.normal([0.0, 0.0], 1.0, (200, 2))
intruders = rng.uniform(-4.0, 4.0, (50, 2))
eval_X = np.vstack([new, intruders])

for name, model in [("fit على نظيف  (novelty)", novelty),
                    ("fit على ملوث (outlier)", outlier)]:
    pred = model.predict(eval_X)
    print(f"{name}: قبول السليمين {((pred[:200] == 1).mean()):.1%} | "
          f"اصطياد الشواذب {((pred[200:] == -1).mean()):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0.0, 8.0, 240)).reshape(-1, 1)
y = np.sin(X.ravel() * 0.9) * 2.0 + rng.normal(0.0, 0.25, 240)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

models = [
    ("SVR", make_pipeline(StandardScaler(),
                          SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=0.1))),
    ("KernelRidge", make_pipeline(StandardScaler(),
                                  KernelRidge(kernel="rbf", gamma=0.1, alpha=0.01))),
]
for name, model in models:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    mse = mean_squared_error(y_te, model.predict(X_te))
    if name == "SVR":
        n_vec = len(model[-1].support_)
    else:
        n_vec = int(model[-1].dual_coef_.size)           # كل عينة تحمل معاملاً
    print(f"{name:>12} | زمن التدريب: {dt * 1000:7.1f} ms | "
          f"MSE: {mse:.4f} | معاملات الحل: {n_vec}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = fetch_california_housing()
X, y = data.data[:1200], data.target[:1200]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'epsilon':>8} | {'متجهات داعمة':>12} | {'R2 الاختبار':>12}")
print("-" * 40)
for eps in [0.01, 0.05, 0.1, 0.3, 0.5]:
    svr = make_pipeline(StandardScaler(),
                        SVR(kernel="rbf", C=10.0, epsilon=eps)).fit(X_tr, y_tr)
    print(f"{eps:>8.2f} | {len(svr[-1].support_):>12} | "
          f"{svr.score(X_te, y_te):>12.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-53-ch53.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

X, y = load_digits(return_X_y=True)
zeros = X[y == 0]
rest = X[y != 0]

print(f"{'gamma':>7} | رفض الرقم 0 | رفض الأرقام الأخرى")
print("-" * 44)
for g in [0.05, 0.01, 0.002]:
    oc = make_pipeline(StandardScaler(),
                       OneClassSVM(kernel="rbf", nu=0.05, gamma=g)).fit(zeros)
    rej0 = (oc.predict(zeros) == -1).mean()
    rejrest = (oc.predict(rest) == -1).mean()
    print(f"{g:>7.3f} | {rej0:>10.1%} | {rejrest:>13.1%}")
