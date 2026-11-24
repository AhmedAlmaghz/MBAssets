#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نماذج النواة والشبكات العصبية
Mastering scikit-learn 1.9 — Chapter 58 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-58-ch58 (lesson 58)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-58-ch58.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نماذج النواة والشبكات العصبية
Mastering scikit-learn 1.9 — Chapter 58 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-58-ch58 (lesson 58)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0, 10, size=(300, 1)), axis=0)
y = np.sin(X[:, 0]) + rng.normal(0, 0.15, size=300)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

linear = Ridge(alpha=1.0).fit(X_tr, y_tr)
kernel = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.5).fit(X_tr, y_tr)
print(f"Ridge الخطي       R² = {linear.score(X_te, y_te):.4f}")
print(f"KernelRidge(rbf)  R² = {kernel.score(X_te, y_te):.4f}")
print(f"طول dual_coef_ = {kernel.dual_coef_.size} = عدد عينات التدريب (لا متجهات داعمة)")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np, time
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = fetch_california_housing()
X, y = data.data[:5000], data.target[:5000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

t0 = time.perf_counter()
kr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.05).fit(X_tr_s, y_tr)
t_kr = time.perf_counter() - t0
t0 = time.perf_counter()
sv = SVR(kernel="rbf", C=10, gamma=0.05).fit(X_tr_s, y_tr)
t_sv = time.perf_counter() - t0

x1 = X_te_s[:1]
t0 = time.perf_counter()
for _ in range(100):
    kr.predict(x1)
tp_kr = (time.perf_counter() - t0) / 100 * 1000
t0 = time.perf_counter()
for _ in range(100):
    sv.predict(x1)
tp_sv = (time.perf_counter() - t0) / 100 * 1000

print(f"{'النموذج':14s} {'R²':>7s} {'تدريب':>7s} {'تنبؤ لعينة':>11s}")
print(f"{'KernelRidge':14s} {kr.score(X_te_s, y_te):7.4f} {t_kr:5.2f}s {tp_kr:9.2f}ms")
print(f"{'SVR':14s} {sv.score(X_te_s, y_te):7.4f} {t_sv:5.2f}s {tp_sv:9.2f}ms")
print(f"معاملات KR: {kr.dual_coef_.size} (كل العينات) | متجهات SVR الداعمة: {sv.support_.size}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=100_000, n_features=12, n_informative=8,
                           n_redundant=2, class_sep=0.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

def bench(name, model):
    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr)
    t = time.perf_counter() - t0
    print(f"{name:30s} الدقة: {model.score(X_te_s, y_te):.4f} | الزمن: {t:5.2f}s")

bench("SGD خطي مباشر", SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1))
bench("Nystroem(64) + SGD", make_pipeline(
    Nystroem(kernel="rbf", n_components=64, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))
bench("Nystroem(200) + SGD", make_pipeline(
    Nystroem(kernel="rbf", n_components=200, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))
bench("RBFSampler(100) + SGD", make_pipeline(
    RBFSampler(gamma=1.0/12, n_components=100, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))

# السقف: SVC بنواة كاملة — مستحيل على 100 ألف، لذا على 6000 فقط
X_small, _, y_small, _ = train_test_split(X_tr_s, y_tr, train_size=6000, random_state=42)
bench("SVC كامل (6000 عينة فقط)", SVC(kernel="rbf", gamma=1.0/12).fit(X_small, y_small))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"اللوجيستي الخطي (سقفه الحد المستقيم): {LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te):.4f}")

for act in ["relu", "tanh"]:
    m = MLPClassifier(hidden_layer_sizes=(32, 16), activation=act, max_iter=2000, random_state=42).fit(X_tr, y_tr)
    print(f"MLP(32,16) تفعيل={act:5s} الدقة: {m.score(X_te, y_te):.4f} | تكرارات: {m.n_iter_}")

m_lb = MLPClassifier(hidden_layer_sizes=(32, 16), solver="lbfgs", max_iter=2000, random_state=42).fit(X_tr, y_tr)
print(f"MLP(32,16) solver=lbfgs الدقة: {m_lb.score(X_te, y_te):.4f} | تكرارات: {m_lb.n_iter_}")

m_es = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, early_stopping=True,
                     validation_fraction=0.15, random_state=42).fit(X_tr, y_tr)
print(f"مع early_stopping: الدقة {m_es.score(X_te, y_te):.4f} | تكرارات: {m_es.n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# الجزء الأول: نفس الشبكة على ميزات متطرفة المقياس، مع تطبيع وبلا تطبيع
rng = np.random.default_rng(42)
n = 900
salary = rng.normal(5200, 1200, size=(n, 1))   # آلاف الدولارات
ratio = rng.normal(0.35, 0.09, size=(n, 1))    # نسبة بين 0 و 1
label = ((salary[:, 0] > 5200) ^ (ratio[:, 0] > 0.35)).astype(int)
X = np.hstack([salary, ratio])
X_tr, X_te, y_tr, y_te = train_test_split(X, label, test_size=0.3, random_state=42, stratify=label)

# نكتم التحذير لأن هدفنا قياس أثر المقياس على الدقة لا حالة التقارب
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42).fit(X_tr, y_tr)
scaler = StandardScaler().fit(X_tr)
scaled = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42).fit(
    np.asarray(scaler.transform(X_tr)), y_tr)
print(f"بلا تطبيع: {raw.score(np.asarray(scaler.transform(X_te)), y_te):.4f} | "
      f"مع StandardScaler: {scaled.score(np.asarray(scaler.transform(X_te)), y_te):.4f}")

# الجزء الثاني: تدريب تزايدي دفعة-دفعة على Digits المحجّمة
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
# np.asarray لضمان مصفوفة رقمية مهما كان transform_output
X_tr_s, X_te_s = np.asarray(sc.transform(X_tr)), np.asarray(sc.transform(X_te))
mlp = MLPClassifier(hidden_layer_sizes=(64,), random_state=42)
classes = np.unique(y)
rng = np.random.default_rng(42)
for epoch in range(10):
    order = rng.permutation(len(X_tr_s))
    for start in range(0, len(X_tr_s), 128):
        batch = order[start:start + 128]
        mlp.partial_fit(X_tr_s[batch], y_tr[batch], classes=classes)
    print(f"الدورة {epoch + 1:2d}: loss_ = {mlp.loss_:.4f}")
print(f"الدقة على الاختبار بعد التدريب التزايدي: {mlp.score(X_te_s, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)
X_bin = (X > 4).astype(float)   # رقمنة: بكسل مضيء (1) أو مطفأ (0)
X_tr, X_te, y_tr, y_te = train_test_split(X_bin, y, test_size=0.3, random_state=42, stratify=y)

direct = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
print(f"اللوجيستي على البكسلات الخام: {direct.score(X_te, y_te):.4f}")

rbm_pipe = make_pipeline(
    BernoulliRBM(n_components=64, learning_rate=0.1, n_iter=30, random_state=42),
    LogisticRegression(max_iter=2000),
)
rbm_pipe.fit(X_tr, y_tr)
print(f"اللوجيستي على خصائص RBM (64):  {rbm_pipe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data[:3000], data.target[:3000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_tr)
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    model = KernelRidge(kernel="rbf", alpha=alpha, gamma=0.05).fit(scaler.transform(X_tr), y_tr)
    rmse = root_mean_squared_error(y_te, model.predict(scaler.transform(X_te)))
    print(f"alpha={alpha:6.3f} | RMSE = {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch58.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)
for hls in [(32,), (64,), (100, 50), (100, 50, 25)]:
    m = MLPClassifier(hidden_layer_sizes=hls, max_iter=600, early_stopping=True,
                      random_state=42).fit(scaler.transform(X_tr), y_tr)
    print(f"hidden_layer_sizes={str(hls):14s} الدقة: {m.score(scaler.transform(X_te), y_te):.4f} | تكرارات: {m.n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.sort(rng.uniform(0, 10, size=(300, 1)), axis=0)
y = np.sin(X[:, 0]) + rng.normal(0, 0.15, size=300)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

linear = Ridge(alpha=1.0).fit(X_tr, y_tr)
kernel = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.5).fit(X_tr, y_tr)
print(f"Ridge الخطي       R² = {linear.score(X_te, y_te):.4f}")
print(f"KernelRidge(rbf)  R² = {kernel.score(X_te, y_te):.4f}")
print(f"طول dual_coef_ = {kernel.dual_coef_.size} = عدد عينات التدريب (لا متجهات داعمة)")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np, time
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = fetch_california_housing()
X, y = data.data[:5000], data.target[:5000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

t0 = time.perf_counter()
kr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.05).fit(X_tr_s, y_tr)
t_kr = time.perf_counter() - t0
t0 = time.perf_counter()
sv = SVR(kernel="rbf", C=10, gamma=0.05).fit(X_tr_s, y_tr)
t_sv = time.perf_counter() - t0

x1 = X_te_s[:1]
t0 = time.perf_counter()
for _ in range(100):
    kr.predict(x1)
tp_kr = (time.perf_counter() - t0) / 100 * 1000
t0 = time.perf_counter()
for _ in range(100):
    sv.predict(x1)
tp_sv = (time.perf_counter() - t0) / 100 * 1000

print(f"{'النموذج':14s} {'R²':>7s} {'تدريب':>7s} {'تنبؤ لعينة':>11s}")
print(f"{'KernelRidge':14s} {kr.score(X_te_s, y_te):7.4f} {t_kr:5.2f}s {tp_kr:9.2f}ms")
print(f"{'SVR':14s} {sv.score(X_te_s, y_te):7.4f} {t_sv:5.2f}s {tp_sv:9.2f}ms")
print(f"معاملات KR: {kr.dual_coef_.size} (كل العينات) | متجهات SVR الداعمة: {sv.support_.size}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=100_000, n_features=12, n_informative=8,
                           n_redundant=2, class_sep=0.8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

def bench(name, model):
    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr)
    t = time.perf_counter() - t0
    print(f"{name:30s} الدقة: {model.score(X_te_s, y_te):.4f} | الزمن: {t:5.2f}s")

bench("SGD خطي مباشر", SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1))
bench("Nystroem(64) + SGD", make_pipeline(
    Nystroem(kernel="rbf", n_components=64, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))
bench("Nystroem(200) + SGD", make_pipeline(
    Nystroem(kernel="rbf", n_components=200, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))
bench("RBFSampler(100) + SGD", make_pipeline(
    RBFSampler(gamma=1.0/12, n_components=100, random_state=42),
    SGDClassifier(loss="hinge", max_iter=100, tol=1e-3, random_state=42, n_jobs=-1)))

# السقف: SVC بنواة كاملة — مستحيل على 100 ألف، لذا على 6000 فقط
X_small, _, y_small, _ = train_test_split(X_tr_s, y_tr, train_size=6000, random_state=42)
bench("SVC كامل (6000 عينة فقط)", SVC(kernel="rbf", gamma=1.0/12).fit(X_small, y_small))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"اللوجيستي الخطي (سقفه الحد المستقيم): {LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te):.4f}")

for act in ["relu", "tanh"]:
    m = MLPClassifier(hidden_layer_sizes=(32, 16), activation=act, max_iter=2000, random_state=42).fit(X_tr, y_tr)
    print(f"MLP(32,16) تفعيل={act:5s} الدقة: {m.score(X_te, y_te):.4f} | تكرارات: {m.n_iter_}")

m_lb = MLPClassifier(hidden_layer_sizes=(32, 16), solver="lbfgs", max_iter=2000, random_state=42).fit(X_tr, y_tr)
print(f"MLP(32,16) solver=lbfgs الدقة: {m_lb.score(X_te, y_te):.4f} | تكرارات: {m_lb.n_iter_}")

m_es = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, early_stopping=True,
                     validation_fraction=0.15, random_state=42).fit(X_tr, y_tr)
print(f"مع early_stopping: الدقة {m_es.score(X_te, y_te):.4f} | تكرارات: {m_es.n_iter_}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# الجزء الأول: نفس الشبكة على ميزات متطرفة المقياس، مع تطبيع وبلا تطبيع
rng = np.random.default_rng(42)
n = 900
salary = rng.normal(5200, 1200, size=(n, 1))   # آلاف الدولارات
ratio = rng.normal(0.35, 0.09, size=(n, 1))    # نسبة بين 0 و 1
label = ((salary[:, 0] > 5200) ^ (ratio[:, 0] > 0.35)).astype(int)
X = np.hstack([salary, ratio])
X_tr, X_te, y_tr, y_te = train_test_split(X, label, test_size=0.3, random_state=42, stratify=label)

# نكتم التحذير لأن هدفنا قياس أثر المقياس على الدقة لا حالة التقارب
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42).fit(X_tr, y_tr)
scaler = StandardScaler().fit(X_tr)
scaled = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42).fit(
    np.asarray(scaler.transform(X_tr)), y_tr)
print(f"بلا تطبيع: {raw.score(np.asarray(scaler.transform(X_te)), y_te):.4f} | "
      f"مع StandardScaler: {scaled.score(np.asarray(scaler.transform(X_te)), y_te):.4f}")

# الجزء الثاني: تدريب تزايدي دفعة-دفعة على Digits المحجّمة
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
# np.asarray لضمان مصفوفة رقمية مهما كان transform_output
X_tr_s, X_te_s = np.asarray(sc.transform(X_tr)), np.asarray(sc.transform(X_te))
mlp = MLPClassifier(hidden_layer_sizes=(64,), random_state=42)
classes = np.unique(y)
rng = np.random.default_rng(42)
for epoch in range(10):
    order = rng.permutation(len(X_tr_s))
    for start in range(0, len(X_tr_s), 128):
        batch = order[start:start + 128]
        mlp.partial_fit(X_tr_s[batch], y_tr[batch], classes=classes)
    print(f"الدورة {epoch + 1:2d}: loss_ = {mlp.loss_:.4f}")
print(f"الدقة على الاختبار بعد التدريب التزايدي: {mlp.score(X_te_s, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)
X_bin = (X > 4).astype(float)   # رقمنة: بكسل مضيء (1) أو مطفأ (0)
X_tr, X_te, y_tr, y_te = train_test_split(X_bin, y, test_size=0.3, random_state=42, stratify=y)

direct = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
print(f"اللوجيستي على البكسلات الخام: {direct.score(X_te, y_te):.4f}")

rbm_pipe = make_pipeline(
    BernoulliRBM(n_components=64, learning_rate=0.1, n_iter=30, random_state=42),
    LogisticRegression(max_iter=2000),
)
rbm_pipe.fit(X_tr, y_tr)
print(f"اللوجيستي على خصائص RBM (64):  {rbm_pipe.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data[:3000], data.target[:3000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_tr)
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    model = KernelRidge(kernel="rbf", alpha=alpha, gamma=0.05).fit(scaler.transform(X_tr), y_tr)
    rmse = root_mean_squared_error(y_te, model.predict(scaler.transform(X_te)))
    print(f"alpha={alpha:6.3f} | RMSE = {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-58-ch58.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)
for hls in [(32,), (64,), (100, 50), (100, 50, 25)]:
    m = MLPClassifier(hidden_layer_sizes=hls, max_iter=600, early_stopping=True,
                      random_state=42).fit(scaler.transform(X_tr), y_tr)
    print(f"hidden_layer_sizes={str(hls):14s} الدقة: {m.score(scaler.transform(X_te), y_te):.4f} | تكرارات: {m.n_iter_}")
