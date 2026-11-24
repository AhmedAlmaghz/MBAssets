#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المصفوفات المتفرّرة وواجهة sparray الجديدة
Mastering scikit-learn 1.9 — sparray

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-11-ch11 (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — المصفوفات المتفرّرة وواجهة sparray الجديدة
Mastering scikit-learn 1.9 — sparray

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-11-ch11 (lesson 11)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(0)
D = rng.random((6, 5))
D[D < 0.6] = 0.0
print("المصفوفة الكثيفة:")
print(np.round(D, 3))

csr = sp.csr_matrix(D)
coo = sp.coo_matrix(D)
lil = sp.lil_matrix(D)

for name, m in [("csr", csr), ("coo", coo), ("lil", lil)]:
    print(f"{name:4s} nnz={m.nnz:3d} shape={m.shape} has getformat={hasattr(m, 'getformat')}")

print("\ncsr.indptr  (أين يبدأ كل صفّ)  :", csr.indptr.tolist())
print("csr.indices (عمود كل قيمة)    :", csr.indices.tolist())
print("csr.data    (القيم نفسها)     :", np.round(csr.data, 3).tolist())
print("coo.row     :", coo.row.tolist())
print("coo.col     :", coo.col.tolist())
print("\nالصفّ 3 في csr:", csr[3].toarray())
print("تحويل ذهاباً وإياباً بلا فقد:", (csr.tolil().tocsr() != csr).nnz == 0)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
n_rows, n_cols = 500, 400
print(f"{'الكثافة':>8} {'nnz':>8} {'كثيف (MB)':>11} {'CSR (KB)':>10} {'التوفير':>9}")
for d in [0.005, 0.02, 0.10, 0.30]:
    M = sp.random(n_rows, n_cols, density=d, format="csr", random_state=0)
    csr = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    dn = n_rows * n_cols * 8
    print(f"{d:8.3f} {M.nnz:8d} {dn / 1024**2:11.2f} {csr / 1024:10.1f} {dn / csr:8.0f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.neighbors import NearestNeighbors

n, p = 3000, 800
X_dense = rng.random((n, p))
X_sparse = sp.random(n, p, density=0.01, format="csr", random_state=0)
q_sparse = sp.random(1, p, density=0.01, format="csr", random_state=1)
q_dense = rng.random((1, p))

t0 = time.perf_counter()
NearestNeighbors(n_neighbors=5).fit(X_sparse).kneighbors(q_sparse)
t_sparse = time.perf_counter() - t0

t0 = time.perf_counter()
NearestNeighbors(n_neighbors=5).fit(X_dense).kneighbors(q_dense)
t_dense = time.perf_counter() - t0

dense_bytes = X_dense.nbytes
sparse_bytes = X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes
print(f"الذاكرة : كثيفة {dense_bytes / 1024**2:.1f} ميغابايت | متفرّرة {sparse_bytes / 1024:.0f} كيلوبايت")
print(f"الزمن    : متفرّرة {t_sparse:.4f}ث | كثيفة {t_dense:.4f}ث | تسريع {t_dense / t_sparse:.0f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

set_config(transform_output="default")
rng2 = np.random.default_rng(0)
n = 600
wide = pd.DataFrame({f"cat{j}": rng2.choice([f"v{k}" for k in range(60)], n) for j in range(40)})
for j in range(5):
    wide[f"num{j}"] = rng2.normal(size=n)
cat_cols = [c for c in wide.columns if c.startswith("cat")]
num_cols = [c for c in wide.columns if c.startswith("num")]

spec = [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
for thr in [0.3, 1.0, 0.0]:
    Z = ColumnTransformer(spec, sparse_threshold=thr).fit_transform(wide)
    dens = Z.nnz / np.prod(Z.shape) if hasattr(Z, "nnz") else np.count_nonzero(Z) / np.prod(Z.shape)
    print(f"  sparse_threshold={thr:<4} -> {type(Z).__name__:11s} shape={Z.shape} density={dens:.4f}")

Z = ColumnTransformer(spec).fit_transform(wide)
sparse_b = Z.data.nbytes + Z.indices.nbytes + Z.indptr.nbytes
print(f"\nلو ضُغطت: {sparse_b / 1024:.0f} كيلوبايت   |   لو كُثّفت: {Z.shape[0] * Z.shape[1] * 8 / 1024**2:.1f} ميغابايت")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config
from sklearn.preprocessing import StandardScaler

# sparse output is incompatible with pandas containers, so go back to arrays
set_config(transform_output="default")

Xsp = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [2.0, 0.0, 1.0]]))

try:
    StandardScaler().fit_transform(Xsp)
except Exception as exc:
    print("with_mean=True  ->", type(exc).__name__, ":", str(exc).splitlines()[0][:96])

out = StandardScaler(with_mean=False).fit_transform(Xsp)
print("with_mean=False ->", type(out).__name__, out.shape, "يعمل")
print("القيم بعد التحويل (مصفوفة أصلاً كانت فيها أصفار):")
print(np.round(out.toarray(), 3))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config, get_config, config_context
from sklearn.preprocessing import OneHotEncoder

# enable the array-style container so sparse output is allowed at all
set_config(transform_output="default")

X_cat = np.array([["a"], ["b"], ["a"], ["c"]], dtype=object)
print("القيمة الافتراضية لـ sparse_interface:", get_config()["sparse_interface"])

for mode in ["spmatrix", "sparray"]:
    set_config(sparse_interface=mode)
    out = OneHotEncoder().fit_transform(X_cat)
    print(f"  sparse_interface={mode:8s} -> {type(out).__name__:11s} shape={out.shape}")

with config_context(sparse_interface="sparray"):
    print("  داخل config_context ->", type(OneHotEncoder().fit_transform(X_cat)).__name__)
print("  بعد الخروج من السياق ->", type(OneHotEncoder().fit_transform(X_cat)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch11.md
# ────────────────────────────────────────────────────────────────────────
dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
m = sp.csr_matrix(dense)
a = sp.csr_array(dense)

print("csr_matrix.getformat() :", m.getformat())
print("csr_array.getformat()  :", hasattr(a, "getformat"), " (السمة .format =", a.format, ")")
print("sp.isspmatrix_csr      :", sp.isspmatrix_csr(m), sp.isspmatrix_csr(a))
print("sp.issparse            :", sp.issparse(m), sp.issparse(a))
print("ndim / shape / nnz      :", (m.ndim, a.ndim), (m.shape, a.shape), (m.nnz, a.nnz))
print("tocsr() تحافظ على الصنف:", type(m.tocsr()).__name__, type(a.tocsr()).__name__)

set_config(sparse_interface="spmatrix")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(0)
D = rng.random((6, 5))
D[D < 0.6] = 0.0
print("المصفوفة الكثيفة:")
print(np.round(D, 3))

csr = sp.csr_matrix(D)
coo = sp.coo_matrix(D)
lil = sp.lil_matrix(D)

for name, m in [("csr", csr), ("coo", coo), ("lil", lil)]:
    print(f"{name:4s} nnz={m.nnz:3d} shape={m.shape} has getformat={hasattr(m, 'getformat')}")

print("\ncsr.indptr  (أين يبدأ كل صفّ)  :", csr.indptr.tolist())
print("csr.indices (عمود كل قيمة)    :", csr.indices.tolist())
print("csr.data    (القيم نفسها)     :", np.round(csr.data, 3).tolist())
print("coo.row     :", coo.row.tolist())
print("coo.col     :", coo.col.tolist())
print("\nالصفّ 3 في csr:", csr[3].toarray())
print("تحويل ذهاباً وإياباً بلا فقد:", (csr.tolil().tocsr() != csr).nnz == 0)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
n_rows, n_cols = 500, 400
print(f"{'الكثافة':>8} {'nnz':>8} {'كثيف (MB)':>11} {'CSR (KB)':>10} {'التوفير':>9}")
for d in [0.005, 0.02, 0.10, 0.30]:
    M = sp.random(n_rows, n_cols, density=d, format="csr", random_state=0)
    csr = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    dn = n_rows * n_cols * 8
    print(f"{d:8.3f} {M.nnz:8d} {dn / 1024**2:11.2f} {csr / 1024:10.1f} {dn / csr:8.0f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.neighbors import NearestNeighbors

n, p = 3000, 800
X_dense = rng.random((n, p))
X_sparse = sp.random(n, p, density=0.01, format="csr", random_state=0)
q_sparse = sp.random(1, p, density=0.01, format="csr", random_state=1)
q_dense = rng.random((1, p))

t0 = time.perf_counter()
NearestNeighbors(n_neighbors=5).fit(X_sparse).kneighbors(q_sparse)
t_sparse = time.perf_counter() - t0

t0 = time.perf_counter()
NearestNeighbors(n_neighbors=5).fit(X_dense).kneighbors(q_dense)
t_dense = time.perf_counter() - t0

dense_bytes = X_dense.nbytes
sparse_bytes = X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes
print(f"الذاكرة : كثيفة {dense_bytes / 1024**2:.1f} ميغابايت | متفرّرة {sparse_bytes / 1024:.0f} كيلوبايت")
print(f"الزمن    : متفرّرة {t_sparse:.4f}ث | كثيفة {t_dense:.4f}ث | تسريع {t_dense / t_sparse:.0f}x")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

set_config(transform_output="default")
rng2 = np.random.default_rng(0)
n = 600
wide = pd.DataFrame({f"cat{j}": rng2.choice([f"v{k}" for k in range(60)], n) for j in range(40)})
for j in range(5):
    wide[f"num{j}"] = rng2.normal(size=n)
cat_cols = [c for c in wide.columns if c.startswith("cat")]
num_cols = [c for c in wide.columns if c.startswith("num")]

spec = [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
for thr in [0.3, 1.0, 0.0]:
    Z = ColumnTransformer(spec, sparse_threshold=thr).fit_transform(wide)
    dens = Z.nnz / np.prod(Z.shape) if hasattr(Z, "nnz") else np.count_nonzero(Z) / np.prod(Z.shape)
    print(f"  sparse_threshold={thr:<4} -> {type(Z).__name__:11s} shape={Z.shape} density={dens:.4f}")

Z = ColumnTransformer(spec).fit_transform(wide)
sparse_b = Z.data.nbytes + Z.indices.nbytes + Z.indptr.nbytes
print(f"\nلو ضُغطت: {sparse_b / 1024:.0f} كيلوبايت   |   لو كُثّفت: {Z.shape[0] * Z.shape[1] * 8 / 1024**2:.1f} ميغابايت")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config
from sklearn.preprocessing import StandardScaler

# sparse output is incompatible with pandas containers, so go back to arrays
set_config(transform_output="default")

Xsp = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [2.0, 0.0, 1.0]]))

try:
    StandardScaler().fit_transform(Xsp)
except Exception as exc:
    print("with_mean=True  ->", type(exc).__name__, ":", str(exc).splitlines()[0][:96])

out = StandardScaler(with_mean=False).fit_transform(Xsp)
print("with_mean=False ->", type(out).__name__, out.shape, "يعمل")
print("القيم بعد التحويل (مصفوفة أصلاً كانت فيها أصفار):")
print(np.round(out.toarray(), 3))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn import set_config, get_config, config_context
from sklearn.preprocessing import OneHotEncoder

# enable the array-style container so sparse output is allowed at all
set_config(transform_output="default")

X_cat = np.array([["a"], ["b"], ["a"], ["c"]], dtype=object)
print("القيمة الافتراضية لـ sparse_interface:", get_config()["sparse_interface"])

for mode in ["spmatrix", "sparray"]:
    set_config(sparse_interface=mode)
    out = OneHotEncoder().fit_transform(X_cat)
    print(f"  sparse_interface={mode:8s} -> {type(out).__name__:11s} shape={out.shape}")

with config_context(sparse_interface="sparray"):
    print("  داخل config_context ->", type(OneHotEncoder().fit_transform(X_cat)).__name__)
print("  بعد الخروج من السياق ->", type(OneHotEncoder().fit_transform(X_cat)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-11-ch11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
m = sp.csr_matrix(dense)
a = sp.csr_array(dense)

print("csr_matrix.getformat() :", m.getformat())
print("csr_array.getformat()  :", hasattr(a, "getformat"), " (السمة .format =", a.format, ")")
print("sp.isspmatrix_csr      :", sp.isspmatrix_csr(m), sp.isspmatrix_csr(a))
print("sp.issparse            :", sp.issparse(m), sp.issparse(a))
print("ndim / shape / nnz      :", (m.ndim, a.ndim), (m.shape, a.shape), (m.nnz, a.nnz))
print("tocsr() تحافظ على الصنف:", type(m.tocsr()).__name__, type(a.tocsr()).__name__)

set_config(sparse_interface="spmatrix")
