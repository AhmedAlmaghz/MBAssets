#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الصور والبيانات غير الجدولية
Mastering scikit-learn 1.9 — Chapter 19 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-19-ch19 (lesson 19)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-19-ch19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الصور والبيانات غير الجدولية
Mastering scikit-learn 1.9 — Chapter 19 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-19-ch19 (lesson 19)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch19.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits

digits = load_digits()

print(f"إجمالي عدد الصور في مجموعة Digits: {digits.images.shape[0]:,} صورة")
print(f"أبعاد مصفوفة الصورة الأصلية (images[0]): {digits.images[0].shape} (شبكة 8×8)")
print(f"أبعاد مصفوفة الخصائص المسطحة (data): {digits.data.shape} (متجه 64 بعداً)")
print("\nعينة من أول 8 بكسلات في المتجه المسطح للصورة الأولى:")
print(digits.data[0][:8].round(1))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch19.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.image import extract_patches_2d

sample_digit = digits.images[0]

# استخراج 8 رقع عشوائية بحجم 3×3 بكسل من صورة الرقم
patches = extract_patches_2d(
    sample_digit,
    patch_size=(3, 3),
    max_patches=8,
    random_state=42
)

print(f"أبعاد مصفوفة الرقع المستخرجة: {patches.shape} (8 رقع بحجم 3×3)")
print("مصفوفة الرقعة الأولى المستخرجة:")
print(patches[0].round(1))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch19.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def extract_gradient_features(images_stack):
    """استخراج ميزات دلالية مكثفة من شبكة البكسلات مباشرة."""
    features_list = []
    for img in images_stack:
        # حساب الفروق المكانية بين البكسلات المتجاورة (مشابه لمرشح سوبل)
        diff_vertical = np.diff(img, axis=0)
        diff_horizontal = np.diff(img, axis=1)

        mean_intensity = np.mean(img)
        var_intensity = np.var(img)
        energy_v = np.mean(diff_vertical ** 2)
        energy_h = np.mean(diff_horizontal ** 2)

        features_list.append([mean_intensity, var_intensity, energy_v, energy_h])
    return np.array(features_list)

engineered_img_feats = extract_gradient_features(digits.images[:4])

print("الميزات الهندسية المستخرجة لأول 4 صور:")
print("الأعمدة: [متوسط الإضاءة، التباين، طاقة التدرج الرأسي، طاقة التدرج الأفقي]")
print(engineered_img_feats.round(2))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch19.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.30, random_state=42, stratify=digits.target
)

# 1. خط أنابيب الانحدار اللوجيستي المنظم
linear_img_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, random_state=42)
)
linear_img_pipe.fit(X_train, y_train)
acc_linear = linear_img_pipe.score(X_test, y_test)

# 2. مصنّف أقرب الجيران (3-NN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
acc_knn = knn_model.score(X_test, y_test)

print("دقة النماذج الكلاسيكية على صور Digits (8×8):")
print(f"  - الانحدار اللوجيستي (مع التحجيم): {acc_linear:.1%}")
print(f"  - أقرب الجيران (3-NN)           : {acc_knn:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch19.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class ImageThresholdBinarizer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=8.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        validate_data(self, X, reset=True)
        return self

    def transform(self, X):
        X = validate_data(self, X, reset=False)
        return (X > self.threshold).astype(float)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-19-ch19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits

digits = load_digits()

print(f"إجمالي عدد الصور في مجموعة Digits: {digits.images.shape[0]:,} صورة")
print(f"أبعاد مصفوفة الصورة الأصلية (images[0]): {digits.images[0].shape} (شبكة 8×8)")
print(f"أبعاد مصفوفة الخصائص المسطحة (data): {digits.data.shape} (متجه 64 بعداً)")
print("\nعينة من أول 8 بكسلات في المتجه المسطح للصورة الأولى:")
print(digits.data[0][:8].round(1))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-19-ch19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.image import extract_patches_2d

sample_digit = digits.images[0]

# استخراج 8 رقع عشوائية بحجم 3×3 بكسل من صورة الرقم
patches = extract_patches_2d(
    sample_digit,
    patch_size=(3, 3),
    max_patches=8,
    random_state=42
)

print(f"أبعاد مصفوفة الرقع المستخرجة: {patches.shape} (8 رقع بحجم 3×3)")
print("مصفوفة الرقعة الأولى المستخرجة:")
print(patches[0].round(1))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-19-ch19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def extract_gradient_features(images_stack):
    """استخراج ميزات دلالية مكثفة من شبكة البكسلات مباشرة."""
    features_list = []
    for img in images_stack:
        # حساب الفروق المكانية بين البكسلات المتجاورة (مشابه لمرشح سوبل)
        diff_vertical = np.diff(img, axis=0)
        diff_horizontal = np.diff(img, axis=1)

        mean_intensity = np.mean(img)
        var_intensity = np.var(img)
        energy_v = np.mean(diff_vertical ** 2)
        energy_h = np.mean(diff_horizontal ** 2)

        features_list.append([mean_intensity, var_intensity, energy_v, energy_h])
    return np.array(features_list)

engineered_img_feats = extract_gradient_features(digits.images[:4])

print("الميزات الهندسية المستخرجة لأول 4 صور:")
print("الأعمدة: [متوسط الإضاءة، التباين، طاقة التدرج الرأسي، طاقة التدرج الأفقي]")
print(engineered_img_feats.round(2))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-19-ch19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.30, random_state=42, stratify=digits.target
)

# 1. خط أنابيب الانحدار اللوجيستي المنظم
linear_img_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, random_state=42)
)
linear_img_pipe.fit(X_train, y_train)
acc_linear = linear_img_pipe.score(X_test, y_test)

# 2. مصنّف أقرب الجيران (3-NN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
acc_knn = knn_model.score(X_test, y_test)

print("دقة النماذج الكلاسيكية على صور Digits (8×8):")
print(f"  - الانحدار اللوجيستي (مع التحجيم): {acc_linear:.1%}")
print(f"  - أقرب الجيران (3-NN)           : {acc_knn:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-19-ch19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class ImageThresholdBinarizer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=8.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        validate_data(self, X, reset=True)
        return self

    def transform(self, X):
        X = validate_data(self, X, reset=False)
        return (X > self.threshold).astype(float)
