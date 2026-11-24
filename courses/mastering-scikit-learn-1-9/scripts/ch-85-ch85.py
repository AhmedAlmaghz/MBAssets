#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ColumnTransformer: تركيب الأعمدة
Mastering scikit-learn 1.9 — ColumnTransformer

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-85-ch85 (lesson 85)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-85-ch85.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ColumnTransformer: تركيب الأعمدة
Mastering scikit-learn 1.9 — ColumnTransformer

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-85-ch85 (lesson 85)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "income": [35_000, 52_000, 41_000, 68_000],
    "visits": [2, 5, 1, 4],
    "city": ["عمّان", "إربد", "عمّان", "الزرقاء"],
})
prep = ColumnTransformer([
    ("num", StandardScaler(), ["income", "visits"]),
    ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
])
Xt = prep.fit_transform(X)
print("شكل المصفوفة:", Xt.shape)
print("الأسماء:", prep.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "age": [21, 42, 36],
    "balance": [120.0, 480.0, 230.0],
    "segment": ["أساسي", "ذهبي", "أساسي"],
})
prep = ColumnTransformer([
    ("numeric", StandardScaler(),
     make_column_selector(dtype_include="number")),
    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     make_column_selector(dtype_include="object")),
])
prep.fit(X)
print("الأعمدة:", prep.feature_names_in_.tolist())
print("الأعمدة الناتجة:", prep.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({
    "age": [20, 40, 60],
    "region_code": [3, 5, 3],
    "record_id": [101, 102, 103],
})
drop_rest = ColumnTransformer(
    [("age", StandardScaler(), ["age"])], remainder="drop")
keep_rest = ColumnTransformer(
    [("age", StandardScaler(), ["age"])], remainder="passthrough")
print("مع الإسقاط:", drop_rest.fit_transform(X).shape)
print("مع المرور:", keep_rest.fit_transform(X).shape)
print("أسماء المرور:", keep_rest.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

X = pd.DataFrame({
    "age": [22, 35, 48, 29, 53, 41, 31, 60],
    "balance": [20, 100, 65, 40, 130, 80, 55, 150],
    "plan": ["A", "B", "A", "A", "B", "B", "A", "B"],
})
y = [0, 1, 1, 0, 1, 1, 0, 1]
numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
prep = ColumnTransformer([("num", numeric, ["age", "balance"]),
                          ("cat", categorical, ["plan"])])
model = Pipeline([("prep", prep), ("clf", LogisticRegression())])
scores = cross_val_score(model, X, y, cv=4, scoring="accuracy")
print("درجات الطيات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({"height": [160, 175, 168], "weight": [58, 80, 65],
                  "team": ["أ", "ب", "أ"]})
ct = ColumnTransformer([
    ("num", StandardScaler(), ["height", "weight"]),
    ("team", OneHotEncoder(sparse_output=False), ["team"]),
])
ct.fit(X)
print(ct.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part10/ch85.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({"value": [10, 20, 30], "id": [700, 701, 702]})
ct = ColumnTransformer([("value", StandardScaler(), ["value"])],
                       remainder="passthrough")
print(ct.fit_transform(X).shape)
print("الأسماء:", ct.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "income": [35_000, 52_000, 41_000, 68_000],
    "visits": [2, 5, 1, 4],
    "city": ["عمّان", "إربد", "عمّان", "الزرقاء"],
})
prep = ColumnTransformer([
    ("num", StandardScaler(), ["income", "visits"]),
    ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
])
Xt = prep.fit_transform(X)
print("شكل المصفوفة:", Xt.shape)
print("الأسماء:", prep.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "age": [21, 42, 36],
    "balance": [120.0, 480.0, 230.0],
    "segment": ["أساسي", "ذهبي", "أساسي"],
})
prep = ColumnTransformer([
    ("numeric", StandardScaler(),
     make_column_selector(dtype_include="number")),
    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     make_column_selector(dtype_include="object")),
])
prep.fit(X)
print("الأعمدة:", prep.feature_names_in_.tolist())
print("الأعمدة الناتجة:", prep.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({
    "age": [20, 40, 60],
    "region_code": [3, 5, 3],
    "record_id": [101, 102, 103],
})
drop_rest = ColumnTransformer(
    [("age", StandardScaler(), ["age"])], remainder="drop")
keep_rest = ColumnTransformer(
    [("age", StandardScaler(), ["age"])], remainder="passthrough")
print("مع الإسقاط:", drop_rest.fit_transform(X).shape)
print("مع المرور:", keep_rest.fit_transform(X).shape)
print("أسماء المرور:", keep_rest.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

X = pd.DataFrame({
    "age": [22, 35, 48, 29, 53, 41, 31, 60],
    "balance": [20, 100, 65, 40, 130, 80, 55, 150],
    "plan": ["A", "B", "A", "A", "B", "B", "A", "B"],
})
y = [0, 1, 1, 0, 1, 1, 0, 1]
numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
prep = ColumnTransformer([("num", numeric, ["age", "balance"]),
                          ("cat", categorical, ["plan"])])
model = Pipeline([("prep", prep), ("clf", LogisticRegression())])
scores = cross_val_score(model, X, y, cv=4, scoring="accuracy")
print("درجات الطيات:", scores.round(3).tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({"height": [160, 175, 168], "weight": [58, 80, 65],
                  "team": ["أ", "ب", "أ"]})
ct = ColumnTransformer([
    ("num", StandardScaler(), ["height", "weight"]),
    ("team", OneHotEncoder(sparse_output=False), ["team"]),
])
ct.fit(X)
print(ct.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-85-ch85.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({"value": [10, 20, 30], "id": [700, 701, 702]})
ct = ColumnTransformer([("value", StandardScaler(), ["value"])],
                       remainder="passthrough")
print(ct.fit_transform(X).shape)
print("الأسماء:", ct.get_feature_names_out().tolist())
