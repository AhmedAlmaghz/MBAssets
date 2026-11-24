#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهة set_output: نتائج على شكل DataFrame
Mastering scikit-learn 1.9 — set_output

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-20-ch20 (lesson 20)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-20-ch20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — واجهة set_output: نتائج على شكل DataFrame
Mastering scikit-learn 1.9 — set_output

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-20-ch20 (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.preprocessing import StandardScaler

# إنشاء جدول بيانات صغير مع فهارس مخصصة
clients_df = pd.DataFrame(
    {"age": [28, 45, 62], "income": [45000, 82000, 115000]},
    index=["عميل_١", "عميل_٢", "عميل_٣"]
)

# 1. المحوّل التقليدي: يعيد مصفوفة NumPy
scaler_np = StandardScaler().set_output(transform="default")
output_np = scaler_np.fit_transform(clients_df)

# 2. المحوّل الحديث: يعيد جدول pandas DataFrame
scaler_pd = StandardScaler().set_output(transform="pandas")
output_pd = scaler_pd.fit_transform(clients_df)

print(f"نوع المخرجات التقليدية (default): {type(output_np).__name__}")
print(f"نوع المخرجات الحديثة (pandas) : {type(output_pd).__name__}")
print("\nالجدول الناتج مع الاحتفاظ بالأسماء والفهارس العربية:")
print(output_pd.round(3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
import sklearn
from sklearn import get_config, set_config
from sklearn.preprocessing import RobustScaler

# فحص الإعداد النشط حالياً في بيئة العمل
current_transform = get_config()["transform_output"]
print(f"الإعداد المركزي النشط لـ transform_output: {current_transform}")

# تجربة محوّل جديد دون ضبط محلي
robust = RobustScaler()
transformed_sample = robust.fit_transform(clients_df)

print("نوع مخرجات المحوّل الجديد تلقائياً:", type(transformed_sample).__name__)
print("أعمدة الجدول المعالج المحفوظة:", list(transformed_sample.columns))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
scaler.set_output(transform="polars")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

employees_df = pd.DataFrame({
    "years_exp": [2, 7, 15],
    "rating": [3.8, 4.5, 4.9],
    "department": ["مبيعات", "هندسة", "مبيعات"]
})

ct_pipeline = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["years_exp", "rating"]),
        ("cat", OneHotEncoder(sparse_output=False), ["department"])
    ]
).set_output(transform="pandas")

processed_df = ct_pipeline.fit_transform(employees_df)

print("أبعاد الجدول المجمّع الناتج:", processed_df.shape)
print("\nأسماء الأعمدة المحدثة تلقائياً في DataFrame:")
for col in processed_df.columns:
    print(f"  -> {col}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

poly_tool = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_tool.fit(clients_df[["age", "income"]])

out_names = poly_tool.get_feature_names_out()
print("أسماء الميزات التي صرّحت بها PolynomialFeatures بعد التدريب:")
for name in out_names:
    print(f"  - {name}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

class RatioCalculator(TransformerMixin, BaseEstimator):
    """محوّل مخصّص يحسب نسبة العمود الأول إلى الثاني ويدعم set_output."""
    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        return self

    def transform(self, X):
        X = validate_data(self, X, reset=False)
        ratio = (X[:, 0] / np.maximum(X[:, 1], 1e-6)).reshape(-1, 1)
        return ratio

    def get_feature_names_out(self, input_features=None):
        return np.array(["computed_ratio"])

# اختبار المحوّل المخصّص مع تفعيل pandas
calc = RatioCalculator().set_output(transform="pandas")
ratio_df = calc.fit_transform(clients_df[["income", "age"]])

print("نوع مخرجات المحوّل المخصّص:", type(ratio_df).__name__)
print("اسم العمود الناتج        :", list(ratio_df.columns))
print("فهرس العينات الأصلي       :", list(ratio_df.index))
print("\nالجدول الناتج من المحوّل المخصّص:")
print(ratio_df.round(2))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch20.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, drop_idx=0):
        self.drop_idx = drop_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.delete(X, self.drop_idx, axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")
        return np.delete(input_features, self.drop_idx)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.preprocessing import StandardScaler

# إنشاء جدول بيانات صغير مع فهارس مخصصة
clients_df = pd.DataFrame(
    {"age": [28, 45, 62], "income": [45000, 82000, 115000]},
    index=["عميل_١", "عميل_٢", "عميل_٣"]
)

# 1. المحوّل التقليدي: يعيد مصفوفة NumPy
scaler_np = StandardScaler().set_output(transform="default")
output_np = scaler_np.fit_transform(clients_df)

# 2. المحوّل الحديث: يعيد جدول pandas DataFrame
scaler_pd = StandardScaler().set_output(transform="pandas")
output_pd = scaler_pd.fit_transform(clients_df)

print(f"نوع المخرجات التقليدية (default): {type(output_np).__name__}")
print(f"نوع المخرجات الحديثة (pandas) : {type(output_pd).__name__}")
print("\nالجدول الناتج مع الاحتفاظ بالأسماء والفهارس العربية:")
print(output_pd.round(3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import sklearn
from sklearn import get_config, set_config
from sklearn.preprocessing import RobustScaler

# فحص الإعداد النشط حالياً في بيئة العمل
current_transform = get_config()["transform_output"]
print(f"الإعداد المركزي النشط لـ transform_output: {current_transform}")

# تجربة محوّل جديد دون ضبط محلي
robust = RobustScaler()
transformed_sample = robust.fit_transform(clients_df)

print("نوع مخرجات المحوّل الجديد تلقائياً:", type(transformed_sample).__name__)
print("أعمدة الجدول المعالج المحفوظة:", list(transformed_sample.columns))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
scaler.set_output(transform="polars")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

employees_df = pd.DataFrame({
    "years_exp": [2, 7, 15],
    "rating": [3.8, 4.5, 4.9],
    "department": ["مبيعات", "هندسة", "مبيعات"]
})

ct_pipeline = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["years_exp", "rating"]),
        ("cat", OneHotEncoder(sparse_output=False), ["department"])
    ]
).set_output(transform="pandas")

processed_df = ct_pipeline.fit_transform(employees_df)

print("أبعاد الجدول المجمّع الناتج:", processed_df.shape)
print("\nأسماء الأعمدة المحدثة تلقائياً في DataFrame:")
for col in processed_df.columns:
    print(f"  -> {col}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

poly_tool = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_tool.fit(clients_df[["age", "income"]])

out_names = poly_tool.get_feature_names_out()
print("أسماء الميزات التي صرّحت بها PolynomialFeatures بعد التدريب:")
for name in out_names:
    print(f"  - {name}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

class RatioCalculator(TransformerMixin, BaseEstimator):
    """محوّل مخصّص يحسب نسبة العمود الأول إلى الثاني ويدعم set_output."""
    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        return self

    def transform(self, X):
        X = validate_data(self, X, reset=False)
        ratio = (X[:, 0] / np.maximum(X[:, 1], 1e-6)).reshape(-1, 1)
        return ratio

    def get_feature_names_out(self, input_features=None):
        return np.array(["computed_ratio"])

# اختبار المحوّل المخصّص مع تفعيل pandas
calc = RatioCalculator().set_output(transform="pandas")
ratio_df = calc.fit_transform(clients_df[["income", "age"]])

print("نوع مخرجات المحوّل المخصّص:", type(ratio_df).__name__)
print("اسم العمود الناتج        :", list(ratio_df.columns))
print("فهرس العينات الأصلي       :", list(ratio_df.index))
print("\nالجدول الناتج من المحوّل المخصّص:")
print(ratio_df.round(2))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-20-ch20.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, drop_idx=0):
        self.drop_idx = drop_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.delete(X, self.drop_idx, axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")
        return np.delete(input_features, self.drop_idx)
