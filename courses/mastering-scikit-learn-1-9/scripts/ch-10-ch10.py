#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنماط البيانات في scikit-learn
Mastering scikit-learn 1.9 — scikit-learn

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-10-ch10 (lesson 10)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-10-ch10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنماط البيانات في scikit-learn
Mastering scikit-learn 1.9 — scikit-learn

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-10-ch10 (lesson 10)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(df[["age", "city"]], y)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
ValueError: could not convert string to float: 'Riyadh'

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.datasets import load_diabetes

dia = load_diabetes()
print(f"نوع الكائن      : {type(dia).__name__}")
print(f"المفاتيح المتاحة: {sorted(k for k in dia if not k.startswith('_'))}")
print(f"شكل X           : {dia.data.shape}  dtype={dia.data.dtype}")
print(f"شكل y           : {dia.target.shape}")
print(f"أسماء الخصائص   : {list(dia.feature_names)}")
print()
dia_df = pd.DataFrame(dia.data, columns=dia.feature_names)
dia_df.head()

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

print("dtype          :", A.dtype)
print("shape          :", A.shape, "->", A.shape[0], "صف و", A.shape[1], "عمود")
print("الصف الثاني    :", A[1])
print("العمود الثالث  :", A[:, 2])
print("قناع منطقي يخفي الصف الأخير:", A[A[:, 0] > 3])
print("ما تراه المكتبة:", type(A).__name__, "— بلا أسماء أعمدة")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "age": [23, 45, 31, 52, 38],
    "city": ["Riyadh", "Cairo", "Riyadh", "Amman", "Cairo"],
    "joined": pd.to_datetime(["2023-01-15", "2021-06-02", "2024-03-20", "2020-11-11", "2022-08-08"]),
    "active": [True, False, True, True, False],
})
print(df.to_string(index=False))
print()
print(df.dtypes)
print()
print(f"حجم الذاكرة المستعمل فعلياً: {df.memory_usage(deep=True).sum()} بايت")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

with_names = pd.DataFrame(dia.data[:60], columns=dia.feature_names)
sc_named = StandardScaler().fit(with_names)
sc_anon = StandardScaler().fit(dia.data[:60])

print("مع DataFrame -> feature_names_in_:", list(sc_named.feature_names_in_[:4]), "…")
print("مع ndarray   -> feature_names_in_:", hasattr(sc_anon, "feature_names_in_"))
print("get_feature_names_out() مع أسماء :", sc_named.get_feature_names_out()[:4])
print("get_feature_names_out() بلا أسماء:", sc_anon.get_feature_names_out()[:4])
print("عدد الخصائص في الحالتين          :", sc_named.n_features_in_, sc_anon.n_features_in_)

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
import scipy.sparse as sp

n_rows, n_cols, density = 2000, 300, 0.02
S = sp.random(n_rows, n_cols, density=density, format="csr", random_state=0)
sparse_bytes = S.data.nbytes + S.indices.nbytes + S.indptr.nbytes
dense_bytes = n_rows * n_cols * 8

print(f"كثافة المصفوفة : {S.nnz / (n_rows * n_cols):.4f}  ({S.nnz} غير صفر من {n_rows * n_cols})")
print(f"حجم ndarray    : {dense_bytes / 1024**2:6.2f} ميغابايت")
print(f"حجم CSR        : {sparse_bytes / 1024:6.1f} كيلوبايت")
print(f"نسبة التوفير   : {dense_bytes / sparse_bytes:6.1f} أضعاف")
print()
print(f"نوع البيانات  : {S.data.dtype}   نوع الفهارس: {S.indices.dtype}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.validation import check_array
from sklearn.linear_model import LogisticRegression

mixed = pd.DataFrame({"age": [23.0, 45.0, 31.0, 52.0],
                      "city": ["Riyadh", "Cairo", "Riyadh", "Amman"]})

for kw in [dict(), dict(dtype=None)]:
    label = kw if kw else "افتراضي"
    try:
        out = check_array(mixed, **kw)
        print(f"check_array(dtype={label}) -> نجح: {type(out).__name__} dtype={out.dtype}")
    except Exception as exc:
        print(f"check_array(dtype={label}) -> {type(exc).__name__}: {str(exc).splitlines()[0][:88]}")

try:
    LogisticRegression(max_iter=500).fit(mixed, [0, 1, 1, 0])
except Exception as exc:
    print(f"LogisticRegression.fit         -> {type(exc).__name__}: {str(exc).splitlines()[0][:88]}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), ["age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ])),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(mixed, [0, 1, 1, 0])
print("التنبّؤ        :", pipe.predict(mixed))
print("أسماء المخرجات:", pipe.named_steps["pre"].get_feature_names_out())

# ────────────────────────────────────────────────────────────────────────
# [010] content/part02/ch10.md
# ────────────────────────────────────────────────────────────────────────
import scipy.sparse as sp

n_rows, n_cols = 300, 50
print(f"{'الكثافة':>8} {'nnz':>7} {'ndarray':>12} {'CSR':>10} {'الأوفر':>9}")
for d in [0.05, 0.10, 0.20, 0.40]:
    M = sp.random(n_rows, n_cols, density=d, format="csr", random_state=0)
    csr = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    dn = n_rows * n_cols * 8
    print(f"{d:8.2f} {M.nnz:7d} {dn:12,d} {csr:10,d} {'CSR' if csr < dn else 'ndarray':>9}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(df[["age", "city"]], y)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
ValueError: could not convert string to float: 'Riyadh'

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.datasets import load_diabetes

dia = load_diabetes()
print(f"نوع الكائن      : {type(dia).__name__}")
print(f"المفاتيح المتاحة: {sorted(k for k in dia if not k.startswith('_'))}")
print(f"شكل X           : {dia.data.shape}  dtype={dia.data.dtype}")
print(f"شكل y           : {dia.target.shape}")
print(f"أسماء الخصائص   : {list(dia.feature_names)}")
print()
dia_df = pd.DataFrame(dia.data, columns=dia.feature_names)
dia_df.head()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

print("dtype          :", A.dtype)
print("shape          :", A.shape, "->", A.shape[0], "صف و", A.shape[1], "عمود")
print("الصف الثاني    :", A[1])
print("العمود الثالث  :", A[:, 2])
print("قناع منطقي يخفي الصف الأخير:", A[A[:, 0] > 3])
print("ما تراه المكتبة:", type(A).__name__, "— بلا أسماء أعمدة")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "age": [23, 45, 31, 52, 38],
    "city": ["Riyadh", "Cairo", "Riyadh", "Amman", "Cairo"],
    "joined": pd.to_datetime(["2023-01-15", "2021-06-02", "2024-03-20", "2020-11-11", "2022-08-08"]),
    "active": [True, False, True, True, False],
})
print(df.to_string(index=False))
print()
print(df.dtypes)
print()
print(f"حجم الذاكرة المستعمل فعلياً: {df.memory_usage(deep=True).sum()} بايت")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

with_names = pd.DataFrame(dia.data[:60], columns=dia.feature_names)
sc_named = StandardScaler().fit(with_names)
sc_anon = StandardScaler().fit(dia.data[:60])

print("مع DataFrame -> feature_names_in_:", list(sc_named.feature_names_in_[:4]), "…")
print("مع ndarray   -> feature_names_in_:", hasattr(sc_anon, "feature_names_in_"))
print("get_feature_names_out() مع أسماء :", sc_named.get_feature_names_out()[:4])
print("get_feature_names_out() بلا أسماء:", sc_anon.get_feature_names_out()[:4])
print("عدد الخصائص في الحالتين          :", sc_named.n_features_in_, sc_anon.n_features_in_)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import scipy.sparse as sp

n_rows, n_cols, density = 2000, 300, 0.02
S = sp.random(n_rows, n_cols, density=density, format="csr", random_state=0)
sparse_bytes = S.data.nbytes + S.indices.nbytes + S.indptr.nbytes
dense_bytes = n_rows * n_cols * 8

print(f"كثافة المصفوفة : {S.nnz / (n_rows * n_cols):.4f}  ({S.nnz} غير صفر من {n_rows * n_cols})")
print(f"حجم ndarray    : {dense_bytes / 1024**2:6.2f} ميغابايت")
print(f"حجم CSR        : {sparse_bytes / 1024:6.1f} كيلوبايت")
print(f"نسبة التوفير   : {dense_bytes / sparse_bytes:6.1f} أضعاف")
print()
print(f"نوع البيانات  : {S.data.dtype}   نوع الفهارس: {S.indices.dtype}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.validation import check_array
from sklearn.linear_model import LogisticRegression

mixed = pd.DataFrame({"age": [23.0, 45.0, 31.0, 52.0],
                      "city": ["Riyadh", "Cairo", "Riyadh", "Amman"]})

for kw in [dict(), dict(dtype=None)]:
    label = kw if kw else "افتراضي"
    try:
        out = check_array(mixed, **kw)
        print(f"check_array(dtype={label}) -> نجح: {type(out).__name__} dtype={out.dtype}")
    except Exception as exc:
        print(f"check_array(dtype={label}) -> {type(exc).__name__}: {str(exc).splitlines()[0][:88]}")

try:
    LogisticRegression(max_iter=500).fit(mixed, [0, 1, 1, 0])
except Exception as exc:
    print(f"LogisticRegression.fit         -> {type(exc).__name__}: {str(exc).splitlines()[0][:88]}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), ["age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ])),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(mixed, [0, 1, 1, 0])
print("التنبّؤ        :", pipe.predict(mixed))
print("أسماء المخرجات:", pipe.named_steps["pre"].get_feature_names_out())

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-10-ch10.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import scipy.sparse as sp

n_rows, n_cols = 300, 50
print(f"{'الكثافة':>8} {'nnz':>7} {'ndarray':>12} {'CSR':>10} {'الأوفر':>9}")
for d in [0.05, 0.10, 0.20, 0.40]:
    M = sp.random(n_rows, n_cols, density=d, format="csr", random_state=0)
    csr = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    dn = n_rows * n_cols * 8
    print(f"{d:8.2f} {M.nnz:7d} {dn:12,d} {csr:10,d} {'CSR' if csr < dn else 'ndarray':>9}")
