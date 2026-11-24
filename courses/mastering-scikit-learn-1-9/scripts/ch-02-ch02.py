#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منظوم بايثون العلمية: أين يقع scikit-learn؟
Mastering scikit-learn 1.9 — scikit-learn

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-02-ch02 (lesson 2)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منظوم بايثون العلمية: أين يقع scikit-learn؟
Mastering scikit-learn 1.9 — scikit-learn

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-02-ch02 (lesson 2)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
import platform
import sys

import scipy
import sklearn

print("python    ", sys.version.split()[0], platform.python_implementation())
print("numpy     ", np.__version__)
print("scipy     ", scipy.__version__)
print("pandas    ", pd.__version__)
print("matplotlib", mpl.__version__)
print("sklearn   ", sklearn.__version__)
print("\nscipy.linalg يوفّر:", [n for n in dir(scipy.linalg)
                              if n in ("solve", "inv", "lstsq", "svd", "qr")])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
A = np.arange(12).reshape(3, 4)

print("A.shape =", A.shape, "  ndim =", A.ndim, "  nbytes =", A.nbytes)
print("A.T.shape =", A.T.shape, "  transpose is a view, not a copy:",
      np.shares_memory(A, A.T))
print("slice A[:, :2] shares memory with A:", np.shares_memory(A, A[:, :2]))

# البثّ: عمود (3,1) + صف (1,4) يعطيان مصفوفة (3,4)
col = A[:, :1]
row = A[:1, :]
print("col", col.shape, "row", row.shape, "-> sum", (col + row).shape)

# المحور 0 يجمع عبر الصفوف فيبقى عدد الأعمدة
print("\nمجموع كل عمود (axis=0):", A.sum(axis=0))
print("مجموع كل صف  (axis=1):", A.sum(axis=1))
print("قاعدة: ينتج في البعد *غير* المذكور")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
from scipy import linalg, optimize, special, stats

xs = np.linspace(-4, 4, 9)
print("special.expit  (logistic, مستقرّة عددياً):", np.round(special.expit(xs), 4))
print("stats.norm.pdf(±1, 0):", np.round(stats.norm.pdf([-1.0, 0.0, 1.0]), 4))
print("stats.norm.cdf(1.96): %.4f" % stats.norm.cdf(1.96))

res = optimize.minimize(lambda t: (np.exp(-t ** 2 / 2) - 0.5) ** 2,
                        x0=0.0, method="Nelder-Mead")
print("optimize.minimize:", res.success, " argmin = %.4f" % res.x[0])

M = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
print("linalg.solve:", np.round(linalg.solve(M, b), 4),
      "   inv(M)@b:", np.round(linalg.inv(M) @ b, 4))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
sales = pd.DataFrame({
    "city": ["أ", "ب", "أ", "ب"],
    "x": [1.0, 2.0, 3.0, 4.0],
    "y": [10, 20, 30, 40],
})
print("shape:", sales.shape, "  dtypes:\n", sales.dtypes.to_string())
print("\nتجميع حسب city:\n", sales.groupby("city")[["x", "y"]].mean().to_string())
print("\nselect_dtypes يختار الأعمدة الرقمية تلقائياً:")
print(sales.select_dtypes("number").columns.tolist())

print("\nالآن NumPy لا يعرف أسماء الأعمدة:")
arr = sales[["x", "y"]].to_numpy()
print("  arr.shape =", arr.shape, "   arr[0] =", arr[0])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
from scipy.sparse import csr_matrix

rng = np.random.default_rng(0)
dense = rng.random((300, 400))
sp = csr_matrix(dense * (dense > 0.97))

sparse_bytes = sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes
print(f"كثافة المصفوفة: {sp.nnz / dense.size:.4f}")
print(f"الحجم الكثيف : {dense.nbytes / 1024:8.1f} KB")
print(f"الحجم المتفرّق: {sparse_bytes / 1024:8.1f} KB")
print(f"نسبة التوفير : {dense.nbytes / sparse_bytes:.1f}x")
print("\nnnz =", sp.nnz, "  من أصل", dense.size)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch02.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris

iris = load_iris()
print("type:", type(iris).__name__)
print("keys:", sorted(iris.keys()))
print()
print("data.shape     :", iris.data.shape, type(iris.data).__name__)
print("target.shape   :", iris.target.shape, " values:", np.unique(iris.target))
print("target_names   :", list(iris.target_names))
print("feature_names  :", iris.feature_names)
print("DESCR line 1   :", iris.DESCR.splitlines()[0])
print("frame (default):", iris.frame)

iris_df = load_iris(as_frame=True)
print("frame (as_frame=True):", type(iris_df.frame).__name__,
      iris_df.frame.shape)
print("target as Series     :", type(iris_df.target).__name__)
print(iris_df.frame.head(3).to_string())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import platform
import sys

import scipy
import sklearn

print("python    ", sys.version.split()[0], platform.python_implementation())
print("numpy     ", np.__version__)
print("scipy     ", scipy.__version__)
print("pandas    ", pd.__version__)
print("matplotlib", mpl.__version__)
print("sklearn   ", sklearn.__version__)
print("\nscipy.linalg يوفّر:", [n for n in dir(scipy.linalg)
                              if n in ("solve", "inv", "lstsq", "svd", "qr")])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
A = np.arange(12).reshape(3, 4)

print("A.shape =", A.shape, "  ndim =", A.ndim, "  nbytes =", A.nbytes)
print("A.T.shape =", A.T.shape, "  transpose is a view, not a copy:",
      np.shares_memory(A, A.T))
print("slice A[:, :2] shares memory with A:", np.shares_memory(A, A[:, :2]))

# البثّ: عمود (3,1) + صف (1,4) يعطيان مصفوفة (3,4)
col = A[:, :1]
row = A[:1, :]
print("col", col.shape, "row", row.shape, "-> sum", (col + row).shape)

# المحور 0 يجمع عبر الصفوف فيبقى عدد الأعمدة
print("\nمجموع كل عمود (axis=0):", A.sum(axis=0))
print("مجموع كل صف  (axis=1):", A.sum(axis=1))
print("قاعدة: ينتج في البعد *غير* المذكور")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from scipy import linalg, optimize, special, stats

xs = np.linspace(-4, 4, 9)
print("special.expit  (logistic, مستقرّة عددياً):", np.round(special.expit(xs), 4))
print("stats.norm.pdf(±1, 0):", np.round(stats.norm.pdf([-1.0, 0.0, 1.0]), 4))
print("stats.norm.cdf(1.96): %.4f" % stats.norm.cdf(1.96))

res = optimize.minimize(lambda t: (np.exp(-t ** 2 / 2) - 0.5) ** 2,
                        x0=0.0, method="Nelder-Mead")
print("optimize.minimize:", res.success, " argmin = %.4f" % res.x[0])

M = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
print("linalg.solve:", np.round(linalg.solve(M, b), 4),
      "   inv(M)@b:", np.round(linalg.inv(M) @ b, 4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
sales = pd.DataFrame({
    "city": ["أ", "ب", "أ", "ب"],
    "x": [1.0, 2.0, 3.0, 4.0],
    "y": [10, 20, 30, 40],
})
print("shape:", sales.shape, "  dtypes:\n", sales.dtypes.to_string())
print("\nتجميع حسب city:\n", sales.groupby("city")[["x", "y"]].mean().to_string())
print("\nselect_dtypes يختار الأعمدة الرقمية تلقائياً:")
print(sales.select_dtypes("number").columns.tolist())

print("\nالآن NumPy لا يعرف أسماء الأعمدة:")
arr = sales[["x", "y"]].to_numpy()
print("  arr.shape =", arr.shape, "   arr[0] =", arr[0])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from scipy.sparse import csr_matrix

rng = np.random.default_rng(0)
dense = rng.random((300, 400))
sp = csr_matrix(dense * (dense > 0.97))

sparse_bytes = sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes
print(f"كثافة المصفوفة: {sp.nnz / dense.size:.4f}")
print(f"الحجم الكثيف : {dense.nbytes / 1024:8.1f} KB")
print(f"الحجم المتفرّق: {sparse_bytes / 1024:8.1f} KB")
print(f"نسبة التوفير : {dense.nbytes / sparse_bytes:.1f}x")
print("\nnnz =", sp.nnz, "  من أصل", dense.size)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-02-ch02.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris

iris = load_iris()
print("type:", type(iris).__name__)
print("keys:", sorted(iris.keys()))
print()
print("data.shape     :", iris.data.shape, type(iris.data).__name__)
print("target.shape   :", iris.target.shape, " values:", np.unique(iris.target))
print("target_names   :", list(iris.target_names))
print("feature_names  :", iris.feature_names)
print("DESCR line 1   :", iris.DESCR.splitlines()[0])
print("frame (default):", iris.frame)

iris_df = load_iris(as_frame=True)
print("frame (as_frame=True):", type(iris_df.frame).__name__,
      iris_df.frame.shape)
print("target as Series     :", type(iris_df.target).__name__)
print(iris_df.frame.head(3).to_string())
