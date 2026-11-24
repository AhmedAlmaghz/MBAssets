#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — هندسة الخصائص: من الصف إلى المعلومة
Mastering scikit-learn 1.9 — Chapter 17 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-17-ch17 (lesson 17)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-17-ch17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — هندسة الخصائص: من الصف إلى المعلومة
Mastering scikit-learn 1.9 — Chapter 17 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-17-ch17 (lesson 17)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# عيّنة بيانات: العمود 0 يمثل عدد الغرف، والعمود 1 يمثل عدد غرف النوم
raw_features = np.array([
    [6.0, 2.0],
    [10.0, 5.0],
    [3.0, 1.0],
    [8.0, 2.0]
])

def add_bedroom_ratio(X):
    # حساب نسبة غرف النوم إلى إجمالي الغرف مع حماية من القسمة على صفر
    rooms = X[:, 0]
    bedrooms = X[:, 1]
    ratio = bedrooms / np.maximum(rooms, 1e-6)
    # دمج النسبة كعمود ثالث جديد
    return np.column_stack([X, ratio])

# إنشاء محوّل قياسي قابل للاستخدام في Pipeline
ratio_transformer = FunctionTransformer(add_bedroom_ratio, validate=True)
transformed_data = ratio_transformer.fit_transform(raw_features)

print("شكل البيانات قبل التحويل:", raw_features.shape)
print("شكل البيانات بعد إضافة النسبة:", transformed_data.shape)
print("\nالبيانات الناتجة (الغرف، النوم، النسبة):")
print(transformed_data.round(3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

# نريد التفاعلات بين الميزات فقط دون رفع الميزات الفردية للأس (interaction_only)
poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

poly_data = poly.fit_transform(raw_features)
feature_names = poly.get_feature_names_out(["rooms", "bedrooms"])

print("عدد الميزات الأصلية :", raw_features.shape[1])
print("عدد الميزات بعد التفاعل:", poly_data.shape[1])
print("أسماء الميزات الجديدة المولدة:", list(feature_names))
print("\nالقيم المولدة للعيّنة الأولى:")
first_row = np.asarray(poly_data)[0]
print(dict(zip(feature_names, first_row.round(2))))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

sample_hours = np.array([0, 6, 12, 18, 23]).reshape(-1, 1)

def encode_cyclical_hours(X):
    # الدورة الزمنية T = 24 ساعة
    sin_feat = np.sin(2 * np.pi * X / 24.0)
    cos_feat = np.cos(2 * np.pi * X / 24.0)
    return np.column_stack([sin_feat, cos_feat])

time_encoder = FunctionTransformer(encode_cyclical_hours, validate=True)
encoded_hours = time_encoder.fit_transform(sample_hours)

print("ساعات اليوم المدخلة:", sample_hours.ravel())
print("\nالترميز الدائري الناتج [sin, cos]:")
for h, (s, c) in zip(sample_hours.ravel(), encoded_hours):
    print(f"  الساعة {h:02d} -> sin: {s:+.3f}, cos: {c:+.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import SplineTransformer

# درجات حرارة تجريبية من 0 إلى 40 درجة مئوية
temperatures = np.array([[5.0], [15.0], [25.0], [35.0]])

# توليد 4 شرائح ناعمة بدرجة تكعيبية (degree=3)
spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
spline_features = spline.fit_transform(temperatures)

print("شكل الميزة الأصلية:", temperatures.shape)
print("شكل الميزات الشرائحية المولدة:", spline_features.shape)
print("\nالتمثيل الشرائحي لدرجة الحرارة 25C:")
row_25c = np.asarray(spline_features)[2]
print(row_25c.round(3))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# إنشاء جدول بيانات مختلط واقعي
df_transactions = pd.DataFrame({
    "amount": [120.50, 45.00, 310.20, 89.90],
    "customer_age": [34, 52, 22, 41],
    "payment_method": ["credit_card", "cash", "credit_card", "crypto"],
    "is_fraud": [0, 0, 1, 0]
})

# بناء معالج الأعمدة بالانتقاء الآلي
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=object))
    ]
)

# تطبيق المعالجة على الميزات المستقلة
X_clean = preprocessor.fit_transform(df_transactions.drop(columns=["is_fraud"]))

print("الأعمدة الأصلية في الجدول:", list(df_transactions.columns))
print("أبعاد المصفوفة المجمعة الناتجة:", X_clean.shape)
print("الميزات بعد المعالجة والتوجيه بنجاح.")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# بيانات دائرية متداخلة يستحيل فصلها بخط مستقيم
X_circ, y_circ = make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=42)

# خط الأساس: مصنّف لوجيستي على البيانات الخام
baseline_score = cross_val_score(LogisticRegression(), X_circ, y_circ, cv=5).mean()

# النموذج المعزز بهندسة الخصائص: إضافة المربعات والتفاعلات عبر PolynomialFeatures
engineered_pipe = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LogisticRegression()
)
engineered_score = cross_val_score(engineered_pipe, X_circ, y_circ, cv=5).mean()

print(f"دقة النموذج الخطي على البيانات الخام      : {baseline_score:.1%}")
print(f"دقة النموذج الخطي بعد إضافة الميزات التفاعلية: {engineered_score:.1%}")
print(f"القفزة في الدقة بفضل هندسة الخصائص فقط     : +{(engineered_score - baseline_score) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part02/ch17.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def euclidean_dist(X, center=(0.0, 0.0)):
    dx = X[:, 0] - center[0]
    dy = X[:, 1] - center[1]
    dist = np.sqrt(dx**2 + dy**2).reshape(-1, 1)
    return np.column_stack([X, dist])

dist_transformer = FunctionTransformer(euclidean_dist, validate=True)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# عيّنة بيانات: العمود 0 يمثل عدد الغرف، والعمود 1 يمثل عدد غرف النوم
raw_features = np.array([
    [6.0, 2.0],
    [10.0, 5.0],
    [3.0, 1.0],
    [8.0, 2.0]
])

def add_bedroom_ratio(X):
    # حساب نسبة غرف النوم إلى إجمالي الغرف مع حماية من القسمة على صفر
    rooms = X[:, 0]
    bedrooms = X[:, 1]
    ratio = bedrooms / np.maximum(rooms, 1e-6)
    # دمج النسبة كعمود ثالث جديد
    return np.column_stack([X, ratio])

# إنشاء محوّل قياسي قابل للاستخدام في Pipeline
ratio_transformer = FunctionTransformer(add_bedroom_ratio, validate=True)
transformed_data = ratio_transformer.fit_transform(raw_features)

print("شكل البيانات قبل التحويل:", raw_features.shape)
print("شكل البيانات بعد إضافة النسبة:", transformed_data.shape)
print("\nالبيانات الناتجة (الغرف، النوم، النسبة):")
print(transformed_data.round(3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

# نريد التفاعلات بين الميزات فقط دون رفع الميزات الفردية للأس (interaction_only)
poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

poly_data = poly.fit_transform(raw_features)
feature_names = poly.get_feature_names_out(["rooms", "bedrooms"])

print("عدد الميزات الأصلية :", raw_features.shape[1])
print("عدد الميزات بعد التفاعل:", poly_data.shape[1])
print("أسماء الميزات الجديدة المولدة:", list(feature_names))
print("\nالقيم المولدة للعيّنة الأولى:")
first_row = np.asarray(poly_data)[0]
print(dict(zip(feature_names, first_row.round(2))))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

sample_hours = np.array([0, 6, 12, 18, 23]).reshape(-1, 1)

def encode_cyclical_hours(X):
    # الدورة الزمنية T = 24 ساعة
    sin_feat = np.sin(2 * np.pi * X / 24.0)
    cos_feat = np.cos(2 * np.pi * X / 24.0)
    return np.column_stack([sin_feat, cos_feat])

time_encoder = FunctionTransformer(encode_cyclical_hours, validate=True)
encoded_hours = time_encoder.fit_transform(sample_hours)

print("ساعات اليوم المدخلة:", sample_hours.ravel())
print("\nالترميز الدائري الناتج [sin, cos]:")
for h, (s, c) in zip(sample_hours.ravel(), encoded_hours):
    print(f"  الساعة {h:02d} -> sin: {s:+.3f}, cos: {c:+.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import SplineTransformer

# درجات حرارة تجريبية من 0 إلى 40 درجة مئوية
temperatures = np.array([[5.0], [15.0], [25.0], [35.0]])

# توليد 4 شرائح ناعمة بدرجة تكعيبية (degree=3)
spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
spline_features = spline.fit_transform(temperatures)

print("شكل الميزة الأصلية:", temperatures.shape)
print("شكل الميزات الشرائحية المولدة:", spline_features.shape)
print("\nالتمثيل الشرائحي لدرجة الحرارة 25C:")
row_25c = np.asarray(spline_features)[2]
print(row_25c.round(3))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# إنشاء جدول بيانات مختلط واقعي
df_transactions = pd.DataFrame({
    "amount": [120.50, 45.00, 310.20, 89.90],
    "customer_age": [34, 52, 22, 41],
    "payment_method": ["credit_card", "cash", "credit_card", "crypto"],
    "is_fraud": [0, 0, 1, 0]
})

# بناء معالج الأعمدة بالانتقاء الآلي
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=object))
    ]
)

# تطبيق المعالجة على الميزات المستقلة
X_clean = preprocessor.fit_transform(df_transactions.drop(columns=["is_fraud"]))

print("الأعمدة الأصلية في الجدول:", list(df_transactions.columns))
print("أبعاد المصفوفة المجمعة الناتجة:", X_clean.shape)
print("الميزات بعد المعالجة والتوجيه بنجاح.")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# بيانات دائرية متداخلة يستحيل فصلها بخط مستقيم
X_circ, y_circ = make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=42)

# خط الأساس: مصنّف لوجيستي على البيانات الخام
baseline_score = cross_val_score(LogisticRegression(), X_circ, y_circ, cv=5).mean()

# النموذج المعزز بهندسة الخصائص: إضافة المربعات والتفاعلات عبر PolynomialFeatures
engineered_pipe = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LogisticRegression()
)
engineered_score = cross_val_score(engineered_pipe, X_circ, y_circ, cv=5).mean()

print(f"دقة النموذج الخطي على البيانات الخام      : {baseline_score:.1%}")
print(f"دقة النموذج الخطي بعد إضافة الميزات التفاعلية: {engineered_score:.1%}")
print(f"القفزة في الدقة بفضل هندسة الخصائص فقط     : +{(engineered_score - baseline_score) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-17-ch17.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def euclidean_dist(X, center=(0.0, 0.0)):
    dx = X[:, 0] - center[0]
    dy = X[:, 1] - center[1]
    dist = np.sqrt(dx**2 + dy**2).reshape(-1, 1)
    return np.column_stack([X, dist])

dist_transformer = FunctionTransformer(euclidean_dist, validate=True)
