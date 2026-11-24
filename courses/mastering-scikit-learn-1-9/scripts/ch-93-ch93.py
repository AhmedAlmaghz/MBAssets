#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — إطار narwhals وتبادل البيانات
Mastering scikit-learn 1.9 — narwhals

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-93-ch93 (lesson 93)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-93-ch93.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — إطار narwhals وتبادل البيانات
Mastering scikit-learn 1.9 — narwhals

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-93-ch93 (lesson 93)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler

frame = pl.read_parquet("customers.parquet")      # DataFrame من polars
df = frame.to_pandas()                             # الحلّ الذي يعرفه الجميع
scaled = StandardScaler().fit_transform(df)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import importlib.util as iu
import importlib.metadata as md

LIBS = ["pandas", "polars", "modin", "pyarrow", "duckdb"]
present = {}
for lib in LIBS:
    try:
        present[lib] = iu.find_spec(lib) is not None
    except (ImportError, ValueError):
        present[lib] = False

for lib, ok in present.items():
    print(f"{lib:>8} : " + ("available" if ok else "MISSING"))

reqs = [r for r in md.requires("scikit-learn") if r.startswith("narwhals")]
print("narwhals requirement:", reqs[0] if reqs else "not declared")
import narwhals
print("installed narwhals :", narwhals.__version__)

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import narwhals
for path in ["sklearn.utils._sklearn_compat",
             "sklearn.utils._set_output"]:
    try:
        __import__(path)
        print(f"{path:>32}: importable")
    except ModuleNotFoundError as exc:
        print(f"{path:>32}: {exc}")
print("narwhals version:", narwhals.__version__)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                           n_redundant=0, class_sep=1.1, random_state=42)
df = pd.DataFrame(X, columns=["age", "income", "tenure", "balance",
                              "score", "region"])
frame = nw.from_native(df, eager_only=True)
print("native  :", type(df).__name__, df.shape)
print("wrapped :", type(frame).__name__, "| backend:", frame.implementation)
print("schema  :", frame.schema)

plain = StandardScaler().set_output(transform="default").fit(df[["income"]])
out = plain.transform(nw.from_native(df[["income"]], eager_only=True))
print("transform(narwhals)  ->", type(out).__name__, out.shape)

wrapped = StandardScaler().fit(df[["income"]])
out_pd = wrapped.transform(nw.from_native(df[["income"]], eager_only=True))
print("transform=pandas     ->", type(out_pd).__name__, list(out_pd.columns))

clf = LogisticRegression(max_iter=5000).fit(df[["age", "income"]], y)
proba = clf.predict_proba(frame[["age", "income"]])
print("predict_proba(nw)    ->", type(proba).__name__, proba.shape)
print("first row            :", np.round(proba[0], 4))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
people = pd.DataFrame({
    "age": [23, 41, 35, 60, 28, 52],
    "income": [1200.0, 5400.0, 3100.0, 900.0, 2200.0, 4300.0],
    "city": ["dubai", "cairo", "dubai", "alex", "cairo", "alex"],
})
nw_people = nw.from_native(people, eager_only=True)

print("متوسط العمر :", nw_people.select(nw.col("age").mean()).item())
print("أعلى دخل    :", nw_people.select(nw.col("income").max()).item())

by_city = (nw_people
           .group_by("city")
           .agg(nw.col("income").mean().alias("mean_income"),
                nw.len().alias("rows"))
           .sort("mean_income", descending=True))
print(by_city.to_native())

rich = (nw_people
        .filter(nw.col("income") > 3000)
        .sort("income", descending=True)
        .select("age", "income"))
print(rich.to_native())

tagged = nw_people.with_columns(
    bracket=nw.when(nw.col("income") >= 3000).then(1).otherwise(0))
print("عدد الفئة   :", tagged.select(nw.col("bracket").sum()).item())
print("backend     :", nw_people.implementation,
      "| to_native:", type(tagged.to_native()).__name__)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline

two = df[["age", "income"]]
pipe_pd = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000))]).fit(two, y)
pipe_nw = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000))])
pipe_nw.fit(nw.from_native(two, eager_only=True), y)

print("fit on pandas   -> feature_names_in_:", hasattr(pipe_pd, "feature_names_in_"))
print("fit on narwhals -> feature_names_in_:", hasattr(pipe_nw, "feature_names_in_"))
print("names from pd :", list(pipe_pd[:-1].get_feature_names_out()))
print("names from nw :", list(pipe_nw[:-1].get_feature_names_out()))
print("تطابق التوقعات:", bool((pipe_pd.predict(two) == pipe_nw.predict(two)).all()))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
print("transform_output العام:", get_config()["transform_output"])

as_default = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=5000))])
as_default.set_output(transform="default").fit(two, y)
print("transform=default ->", type(as_default[:-1].transform(two)).__name__)

as_pandas = Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))])
as_pandas.set_output(transform="pandas").fit(two, y)
out = as_pandas[:-1].transform(two)
print("transform=pandas  ->", type(out).__name__, list(out.columns))
print("step alone        ->", type(as_pandas.named_steps["sc"].transform(two)).__name__)
print("الإعداد العام     ->", get_config()["transform_output"])

# ────────────────────────────────────────────────────────────────────────
# [008] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import time
rng = np.random.default_rng(0)
big = pd.DataFrame(rng.normal(size=(80000, 8)),
                   columns=[f"c{i}" for i in range(8)])
sc = StandardScaler().set_output(transform="default")
sc.fit(big)
big_nw = nw.from_native(big, eager_only=True)

def median_ms(fn, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

t_pd = median_ms(lambda: sc.transform(big))
t_nw = median_ms(lambda: sc.transform(big_nw))
print(f"transform على pandas   : {t_pd:.1f} ms")
print(f"transform على narwhals : {t_nw:.1f} ms")
print(f"النسبة                : {t_nw / t_pd:.2f}")
t_to = median_ms(lambda: nw.to_native(nw.from_native(big, eager_only=True)))
print(f"from_native + to_native: {t_to:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
for target in ["polars", "pyarrow", "modin"]:
    sc = StandardScaler()
    try:
        sc.set_output(transform=target)
    except ValueError as exc:
        print(f"{target:>8}: ValueError -> {str(exc).splitlines()[-1][:88]}")
        continue
    try:
        sc.fit(two).transform(two)
        print(f"{target:>8}: transform worked")
    except Exception as exc:
        print(f"{target:>8}: {type(exc).__name__} -> "
              f"{str(exc).splitlines()[-1][:92]}")

# ────────────────────────────────────────────────────────────────────────
# [010] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df2 = df.copy()
df2["region"] = np.where(df2["income"] > df2["income"].median(), "high", "low")
nw2 = nw.from_native(df2, eager_only=True)

try:
    ColumnTransformer([("num", StandardScaler(), ["age", "income"])]).fit(nw2, y)
    print("by name   -> ok")
except ValueError as exc:
    print("by name   ->", str(exc).splitlines()[0][:88])

by_index = ColumnTransformer([("num", StandardScaler(), [0, 3])]).fit(nw2, y)
print("by index  ->", type(by_index.transform(nw2)).__name__,
      by_index.transform(nw2).shape)

from sklearn.compose import make_column_selector
try:
    ColumnTransformer([("num", StandardScaler(),
                       make_column_selector(dtype_include="number"))]).fit(nw2, y)
    print("selector  -> ok")
except ValueError as exc:
    print("selector  ->", str(exc).splitlines()[0][:88])

# ────────────────────────────────────────────────────────────────────────
# [011] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
pipe = make_pipeline(StandardScaler())
pipe.set_output(transform="pyarrow")
pipe.fit(df)
pipe.transform(df)

# ────────────────────────────────────────────────────────────────────────
# [012] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=["age", "tenure", "balance", "score"])
df["region"] = np.where(df["score"] > 0, "north", "south")
df["region"] = pd.Categorical(df["region"], categories=["north", "south"])
nwdf = nw.from_native(df, eager_only=True)

print("schema:", dict(nwdf.schema)["region"])
print("الأعمدة:", nwdf.columns)
counts = nwdf.group_by("region").agg(nw.len().alias("rows")).sort("region")
print(counts.to_native())

try:
    ColumnTransformer([("num", StandardScaler(), ["age", "balance"])]).fit(nwdf, y)
    print("ColumnTransformer by name: ok")
except ValueError as exc:
    print("by name   ->", str(exc).splitlines()[0][:92])
fixed = ColumnTransformer([("num", StandardScaler(), [0, 3])]).fit(nwdf, y)
print("by index  ->", type(fixed.transform(nwdf)).__name__, fixed.transform(nwdf).shape)
clf = LogisticRegression(max_iter=5000).fit(fixed.transform(nwdf), y)
print("degree:", clf.coef_.shape)

# ────────────────────────────────────────────────────────────────────────
# [013] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import narwhals.stable.v1 as nw
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

two = df[["age", "tenure"]]
plain = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
plain.set_output(transform="default").fit(two, y)
print("default:", type(plain[:-1].transform(two)).__name__)

named = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
named.set_output(transform="pandas").fit(two, y)
out = named[:-1].transform(two)
print("pandas :", type(out).__name__, list(out.columns))
print("أول صفان:", np.round(out.to_numpy()[:2], 4).tolist())
print("global :", get_config()["transform_output"])
print("step   :", type(named.named_steps["sc"].transform(two)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [014] content/part11/ch93.md
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
from sklearn.preprocessing import StandardScaler

for target in ["polars", "pyarrow", "modin", "pandas"]:
    sc = StandardScaler()
    set_at = "ok"
    try:
        sc.set_output(transform=target)
    except Exception as exc:
        set_at = f"{type(exc).__name__} at set_output"
    ran = "-"
    if set_at == "ok":
        try:
            sc.fit(df[["age"]]).transform(df[["age"]])
            ran = "transform ok"
        except Exception as exc:
            ran = f"{type(exc).__name__} at transform"
    print(f"{target:>8}: set_output={set_at:<24} -> {ran}")
print("مدعوم رسمياً: default, pandas, polars")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler

frame = pl.read_parquet("customers.parquet")      # DataFrame من polars
df = frame.to_pandas()                             # الحلّ الذي يعرفه الجميع
scaled = StandardScaler().fit_transform(df)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import importlib.util as iu
import importlib.metadata as md

LIBS = ["pandas", "polars", "modin", "pyarrow", "duckdb"]
present = {}
for lib in LIBS:
    try:
        present[lib] = iu.find_spec(lib) is not None
    except (ImportError, ValueError):
        present[lib] = False

for lib, ok in present.items():
    print(f"{lib:>8} : " + ("available" if ok else "MISSING"))

reqs = [r for r in md.requires("scikit-learn") if r.startswith("narwhals")]
print("narwhals requirement:", reqs[0] if reqs else "not declared")
import narwhals
print("installed narwhals :", narwhals.__version__)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import narwhals
for path in ["sklearn.utils._sklearn_compat",
             "sklearn.utils._set_output"]:
    try:
        __import__(path)
        print(f"{path:>32}: importable")
    except ModuleNotFoundError as exc:
        print(f"{path:>32}: {exc}")
print("narwhals version:", narwhals.__version__)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                           n_redundant=0, class_sep=1.1, random_state=42)
df = pd.DataFrame(X, columns=["age", "income", "tenure", "balance",
                              "score", "region"])
frame = nw.from_native(df, eager_only=True)
print("native  :", type(df).__name__, df.shape)
print("wrapped :", type(frame).__name__, "| backend:", frame.implementation)
print("schema  :", frame.schema)

plain = StandardScaler().set_output(transform="default").fit(df[["income"]])
out = plain.transform(nw.from_native(df[["income"]], eager_only=True))
print("transform(narwhals)  ->", type(out).__name__, out.shape)

wrapped = StandardScaler().fit(df[["income"]])
out_pd = wrapped.transform(nw.from_native(df[["income"]], eager_only=True))
print("transform=pandas     ->", type(out_pd).__name__, list(out_pd.columns))

clf = LogisticRegression(max_iter=5000).fit(df[["age", "income"]], y)
proba = clf.predict_proba(frame[["age", "income"]])
print("predict_proba(nw)    ->", type(proba).__name__, proba.shape)
print("first row            :", np.round(proba[0], 4))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
people = pd.DataFrame({
    "age": [23, 41, 35, 60, 28, 52],
    "income": [1200.0, 5400.0, 3100.0, 900.0, 2200.0, 4300.0],
    "city": ["dubai", "cairo", "dubai", "alex", "cairo", "alex"],
})
nw_people = nw.from_native(people, eager_only=True)

print("متوسط العمر :", nw_people.select(nw.col("age").mean()).item())
print("أعلى دخل    :", nw_people.select(nw.col("income").max()).item())

by_city = (nw_people
           .group_by("city")
           .agg(nw.col("income").mean().alias("mean_income"),
                nw.len().alias("rows"))
           .sort("mean_income", descending=True))
print(by_city.to_native())

rich = (nw_people
        .filter(nw.col("income") > 3000)
        .sort("income", descending=True)
        .select("age", "income"))
print(rich.to_native())

tagged = nw_people.with_columns(
    bracket=nw.when(nw.col("income") >= 3000).then(1).otherwise(0))
print("عدد الفئة   :", tagged.select(nw.col("bracket").sum()).item())
print("backend     :", nw_people.implementation,
      "| to_native:", type(tagged.to_native()).__name__)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline

two = df[["age", "income"]]
pipe_pd = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000))]).fit(two, y)
pipe_nw = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000))])
pipe_nw.fit(nw.from_native(two, eager_only=True), y)

print("fit on pandas   -> feature_names_in_:", hasattr(pipe_pd, "feature_names_in_"))
print("fit on narwhals -> feature_names_in_:", hasattr(pipe_nw, "feature_names_in_"))
print("names from pd :", list(pipe_pd[:-1].get_feature_names_out()))
print("names from nw :", list(pipe_nw[:-1].get_feature_names_out()))
print("تطابق التوقعات:", bool((pipe_pd.predict(two) == pipe_nw.predict(two)).all()))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
print("transform_output العام:", get_config()["transform_output"])

as_default = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=5000))])
as_default.set_output(transform="default").fit(two, y)
print("transform=default ->", type(as_default[:-1].transform(two)).__name__)

as_pandas = Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))])
as_pandas.set_output(transform="pandas").fit(two, y)
out = as_pandas[:-1].transform(two)
print("transform=pandas  ->", type(out).__name__, list(out.columns))
print("step alone        ->", type(as_pandas.named_steps["sc"].transform(two)).__name__)
print("الإعداد العام     ->", get_config()["transform_output"])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import time
rng = np.random.default_rng(0)
big = pd.DataFrame(rng.normal(size=(80000, 8)),
                   columns=[f"c{i}" for i in range(8)])
sc = StandardScaler().set_output(transform="default")
sc.fit(big)
big_nw = nw.from_native(big, eager_only=True)

def median_ms(fn, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

t_pd = median_ms(lambda: sc.transform(big))
t_nw = median_ms(lambda: sc.transform(big_nw))
print(f"transform على pandas   : {t_pd:.1f} ms")
print(f"transform على narwhals : {t_nw:.1f} ms")
print(f"النسبة                : {t_nw / t_pd:.2f}")
t_to = median_ms(lambda: nw.to_native(nw.from_native(big, eager_only=True)))
print(f"from_native + to_native: {t_to:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
for target in ["polars", "pyarrow", "modin"]:
    sc = StandardScaler()
    try:
        sc.set_output(transform=target)
    except ValueError as exc:
        print(f"{target:>8}: ValueError -> {str(exc).splitlines()[-1][:88]}")
        continue
    try:
        sc.fit(two).transform(two)
        print(f"{target:>8}: transform worked")
    except Exception as exc:
        print(f"{target:>8}: {type(exc).__name__} -> "
              f"{str(exc).splitlines()[-1][:92]}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df2 = df.copy()
df2["region"] = np.where(df2["income"] > df2["income"].median(), "high", "low")
nw2 = nw.from_native(df2, eager_only=True)

try:
    ColumnTransformer([("num", StandardScaler(), ["age", "income"])]).fit(nw2, y)
    print("by name   -> ok")
except ValueError as exc:
    print("by name   ->", str(exc).splitlines()[0][:88])

by_index = ColumnTransformer([("num", StandardScaler(), [0, 3])]).fit(nw2, y)
print("by index  ->", type(by_index.transform(nw2)).__name__,
      by_index.transform(nw2).shape)

from sklearn.compose import make_column_selector
try:
    ColumnTransformer([("num", StandardScaler(),
                       make_column_selector(dtype_include="number"))]).fit(nw2, y)
    print("selector  -> ok")
except ValueError as exc:
    print("selector  ->", str(exc).splitlines()[0][:88])

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
pipe = make_pipeline(StandardScaler())
pipe.set_output(transform="pyarrow")
pipe.fit(df)
pipe.transform(df)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=["age", "tenure", "balance", "score"])
df["region"] = np.where(df["score"] > 0, "north", "south")
df["region"] = pd.Categorical(df["region"], categories=["north", "south"])
nwdf = nw.from_native(df, eager_only=True)

print("schema:", dict(nwdf.schema)["region"])
print("الأعمدة:", nwdf.columns)
counts = nwdf.group_by("region").agg(nw.len().alias("rows")).sort("region")
print(counts.to_native())

try:
    ColumnTransformer([("num", StandardScaler(), ["age", "balance"])]).fit(nwdf, y)
    print("ColumnTransformer by name: ok")
except ValueError as exc:
    print("by name   ->", str(exc).splitlines()[0][:92])
fixed = ColumnTransformer([("num", StandardScaler(), [0, 3])]).fit(nwdf, y)
print("by index  ->", type(fixed.transform(nwdf)).__name__, fixed.transform(nwdf).shape)
clf = LogisticRegression(max_iter=5000).fit(fixed.transform(nwdf), y)
print("degree:", clf.coef_.shape)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import narwhals.stable.v1 as nw
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

two = df[["age", "tenure"]]
plain = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
plain.set_output(transform="default").fit(two, y)
print("default:", type(plain[:-1].transform(two)).__name__)

named = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
named.set_output(transform="pandas").fit(two, y)
out = named[:-1].transform(two)
print("pandas :", type(out).__name__, list(out.columns))
print("أول صفان:", np.round(out.to_numpy()[:2], 4).tolist())
print("global :", get_config()["transform_output"])
print("step   :", type(named.named_steps["sc"].transform(two)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-93-ch93.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import narwhals.stable.v1 as nw
from sklearn.preprocessing import StandardScaler

for target in ["polars", "pyarrow", "modin", "pandas"]:
    sc = StandardScaler()
    set_at = "ok"
    try:
        sc.set_output(transform=target)
    except Exception as exc:
        set_at = f"{type(exc).__name__} at set_output"
    ran = "-"
    if set_at == "ok":
        try:
            sc.fit(df[["age"]]).transform(df[["age"]])
            ran = "transform ok"
        except Exception as exc:
            ran = f"{type(exc).__name__} at transform"
    print(f"{target:>8}: set_output={set_at:<24} -> {ran}")
print("مدعوم رسمياً: default, pandas, polars")
