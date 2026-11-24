#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنواع البيانات: عددي، وفئوي، وزمني، ونصي
Mastering scikit-learn 1.9 — Chapter 12 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-12-ch12 (lesson 12)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-12-ch12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أنواع البيانات: عددي، وفئوي، وزمني، ونصي
Mastering scikit-learn 1.9 — Chapter 12 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-12-ch12 (lesson 12)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200
frame = pd.DataFrame({
    "age": rng.integers(18, 80, n),
    "income": rng.lognormal(10, 1.1, n),
    "city": rng.choice(["Riyadh", "Cairo", "Amman", "Doha"], n),
    "plan": rng.choice(["free", "pro", "enterprise"], n),
    "signup": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    "note": rng.choice(["good service", "bad support", "fast delivery", "slow app"], n),
    "active": rng.random(n) > 0.4,
})
print(frame.head(4).to_string(index=False))
print()
print(frame.dtypes)
print()
print(f"حجم الجدول: {frame.memory_usage(deep=True).sum():,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, RobustScaler

num = frame[["age", "income"]]
print("قبل:  income   min %8.0f  max %10.0f" % (num["income"].min(), num["income"].max()))

sc = StandardScaler().fit(num)
z = sc.transform(num).to_numpy()[:, 1]
print("z-score من StandardScaler : الوسيط %.2f | الطرف الأعلى %.2f" % (np.median(z), z.max()))

polluted = num.copy()
polluted.loc[0:2, "income"] = polluted["income"].max() * 40
rz = RobustScaler().fit(polluted[["income"]])
rs = rz.transform(polluted[["income"]]).to_numpy()[:, 0]
sz = StandardScaler().fit(polluted[["income"]]).transform(polluted[["income"]]).to_numpy()[:, 0]
print("مع الشواذ: Robust   الوسيط %7.0f  الطرف %9.1f" % (rz.center_[0], rs.max()))
print("مع الشواذ: Standard الوسيط %7.0f  الطرف %9.1f" % (
    StandardScaler().fit(polluted[["income"]]).mean_[0], sz.max()))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cats = frame[["city", "plan"]]

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cats)
print(f"OneHotEncoder -> {ohe.transform(cats).shape[1]} عموداً")
print("  الأسماء:", list(ohe.get_feature_names_out()))

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats)
print(f"OrdinalEncoder -> {oe.transform(cats).shape[1]} عموداً")
print("  الفئات:", [c.tolist() for c in oe.categories_])
print("  أول ٦ قيم:", oe.transform(cats).to_numpy()[:6, 0].ravel().astype(int))

rare = pd.DataFrame({"city": ["Riyadh"] * 60 + ["Cairo"] * 20 + ["Amman"] * 12 + ["Doha"] * 8})
print("\nتوزيع المدينة قبل التجميع:", rare["city"].value_counts().to_dict())
folded = OneHotEncoder(min_frequency=15, sparse_output=False,
                       handle_unknown="infrequent_if_exist").fit(rare[["city"]])
print("بعد min_frequency=15:", list(folded.get_feature_names_out()))
plain = OneHotEncoder(sparse_output=False).fit(rare[["city"]])
print("بدون min_frequency :", len(plain.get_feature_names_out()), "عموداً")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def date_parts(d):
    s = d.iloc[:, 0] if hasattr(d, "iloc") else d
    month = s.dt.month
    return pd.DataFrame({
        "year": s.dt.year,
        "month": month,
        "dow": s.dt.dayofweek,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    })


ct_dt = ColumnTransformer([("dt", FunctionTransformer(date_parts), ["signup"])])
print("مخرجات مسار الزمن:")
print(ct_dt.fit_transform(frame.head(5)).round(3))
print()
print("أسماء المخرجات:", list(ct_dt.get_feature_names_out()))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(max_features=4)
M = tv.fit_transform(frame["note"])
print("نوع المخرجات:", type(M).__name__, " الشكل:", M.shape)
print("المفردات:", list(tv.get_feature_names_out()))
print("\nأول ٤ صفوف كمصفوفة كثيفة:")
print(pd.DataFrame(M.toarray(), columns=tv.get_feature_names_out()).head(4).round(3).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

for kw in [dict(dtype_include="number"), dict(dtype_include="string"),
           dict(dtype_include="datetime"), dict(dtype_include=bool),
           dict(dtype_exclude="object"), dict(pattern="^(city|plan)$")]:
    print(f"  {str(kw):45s} -> {make_column_selector(**kw)(frame)}")

print("\n--- والآن نحوّل هذه القراءة إلى خطّ فعلي ---")
ct_auto = ColumnTransformer([
    ("num", StandardScaler(), make_column_selector(dtype_include="number")),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     make_column_selector(dtype_include="string", pattern="^(city|plan)$")),
])

Z = ct_auto.fit_transform(frame)
print("نوع الناتج:", type(Z).__name__, " الشكل:", Z.shape)
print("أعمدة المسار الرقمي :", [c for c in ct_auto.get_feature_names_out() if c.startswith("num")])
print("أعمدة المسار الفئوي :", [c for c in ct_auto.get_feature_names_out() if c.startswith("cat")])
print("أعمدة مُهمَلة بصمت   :", ["note", "signup", "active"])

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch12.md
# ────────────────────────────────────────────────────────────────────────
raw = pd.DataFrame({"quantity": ["12", "7", "unknown", "30", " 5 "]})
print("النوع الأصلي:", raw["quantity"].dtype)
clean = pd.to_numeric(raw["quantity"], errors="coerce")
print("بعد to_numeric:", clean.dtype, "| قيم NaN:", int(clean.isna().sum()))
print("لماذا هذا أفضل من الخطأ الصامت؟ لأن الصفّ الثالث صار معلومة مفقودة")
print("يمكن دراستها، بدل أن تتحوّل إلى NaN في صمت أو يُرفض المشروع كله.")
print()
print(raw.assign(quantity=clean).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200
frame = pd.DataFrame({
    "age": rng.integers(18, 80, n),
    "income": rng.lognormal(10, 1.1, n),
    "city": rng.choice(["Riyadh", "Cairo", "Amman", "Doha"], n),
    "plan": rng.choice(["free", "pro", "enterprise"], n),
    "signup": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    "note": rng.choice(["good service", "bad support", "fast delivery", "slow app"], n),
    "active": rng.random(n) > 0.4,
})
print(frame.head(4).to_string(index=False))
print()
print(frame.dtypes)
print()
print(f"حجم الجدول: {frame.memory_usage(deep=True).sum():,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, RobustScaler

num = frame[["age", "income"]]
print("قبل:  income   min %8.0f  max %10.0f" % (num["income"].min(), num["income"].max()))

sc = StandardScaler().fit(num)
z = sc.transform(num).to_numpy()[:, 1]
print("z-score من StandardScaler : الوسيط %.2f | الطرف الأعلى %.2f" % (np.median(z), z.max()))

polluted = num.copy()
polluted.loc[0:2, "income"] = polluted["income"].max() * 40
rz = RobustScaler().fit(polluted[["income"]])
rs = rz.transform(polluted[["income"]]).to_numpy()[:, 0]
sz = StandardScaler().fit(polluted[["income"]]).transform(polluted[["income"]]).to_numpy()[:, 0]
print("مع الشواذ: Robust   الوسيط %7.0f  الطرف %9.1f" % (rz.center_[0], rs.max()))
print("مع الشواذ: Standard الوسيط %7.0f  الطرف %9.1f" % (
    StandardScaler().fit(polluted[["income"]]).mean_[0], sz.max()))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cats = frame[["city", "plan"]]

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cats)
print(f"OneHotEncoder -> {ohe.transform(cats).shape[1]} عموداً")
print("  الأسماء:", list(ohe.get_feature_names_out()))

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats)
print(f"OrdinalEncoder -> {oe.transform(cats).shape[1]} عموداً")
print("  الفئات:", [c.tolist() for c in oe.categories_])
print("  أول ٦ قيم:", oe.transform(cats).to_numpy()[:6, 0].ravel().astype(int))

rare = pd.DataFrame({"city": ["Riyadh"] * 60 + ["Cairo"] * 20 + ["Amman"] * 12 + ["Doha"] * 8})
print("\nتوزيع المدينة قبل التجميع:", rare["city"].value_counts().to_dict())
folded = OneHotEncoder(min_frequency=15, sparse_output=False,
                       handle_unknown="infrequent_if_exist").fit(rare[["city"]])
print("بعد min_frequency=15:", list(folded.get_feature_names_out()))
plain = OneHotEncoder(sparse_output=False).fit(rare[["city"]])
print("بدون min_frequency :", len(plain.get_feature_names_out()), "عموداً")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def date_parts(d):
    s = d.iloc[:, 0] if hasattr(d, "iloc") else d
    month = s.dt.month
    return pd.DataFrame({
        "year": s.dt.year,
        "month": month,
        "dow": s.dt.dayofweek,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    })


ct_dt = ColumnTransformer([("dt", FunctionTransformer(date_parts), ["signup"])])
print("مخرجات مسار الزمن:")
print(ct_dt.fit_transform(frame.head(5)).round(3))
print()
print("أسماء المخرجات:", list(ct_dt.get_feature_names_out()))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(max_features=4)
M = tv.fit_transform(frame["note"])
print("نوع المخرجات:", type(M).__name__, " الشكل:", M.shape)
print("المفردات:", list(tv.get_feature_names_out()))
print("\nأول ٤ صفوف كمصفوفة كثيفة:")
print(pd.DataFrame(M.toarray(), columns=tv.get_feature_names_out()).head(4).round(3).to_string(index=False))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

for kw in [dict(dtype_include="number"), dict(dtype_include="string"),
           dict(dtype_include="datetime"), dict(dtype_include=bool),
           dict(dtype_exclude="object"), dict(pattern="^(city|plan)$")]:
    print(f"  {str(kw):45s} -> {make_column_selector(**kw)(frame)}")

print("\n--- والآن نحوّل هذه القراءة إلى خطّ فعلي ---")
ct_auto = ColumnTransformer([
    ("num", StandardScaler(), make_column_selector(dtype_include="number")),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     make_column_selector(dtype_include="string", pattern="^(city|plan)$")),
])

Z = ct_auto.fit_transform(frame)
print("نوع الناتج:", type(Z).__name__, " الشكل:", Z.shape)
print("أعمدة المسار الرقمي :", [c for c in ct_auto.get_feature_names_out() if c.startswith("num")])
print("أعمدة المسار الفئوي :", [c for c in ct_auto.get_feature_names_out() if c.startswith("cat")])
print("أعمدة مُهمَلة بصمت   :", ["note", "signup", "active"])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-12-ch12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
raw = pd.DataFrame({"quantity": ["12", "7", "unknown", "30", " 5 "]})
print("النوع الأصلي:", raw["quantity"].dtype)
clean = pd.to_numeric(raw["quantity"], errors="coerce")
print("بعد to_numeric:", clean.dtype, "| قيم NaN:", int(clean.isna().sum()))
print("لماذا هذا أفضل من الخطأ الصامت؟ لأن الصفّ الثالث صار معلومة مفقودة")
print("يمكن دراستها، بدل أن تتحوّل إلى NaN في صمت أو يُرفض المشروع كله.")
print()
print(raw.assign(quantity=clean).to_string(index=False))
