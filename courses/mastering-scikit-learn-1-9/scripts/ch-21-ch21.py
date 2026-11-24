#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أبعاد البيانات: لماذا تكره الخوارزمياتmany الأبعاد؟
Mastering scikit-learn 1.9 — many

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-21-ch21 (lesson 21)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-21-ch21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أبعاد البيانات: لماذا تكره الخوارزمياتmany الأبعاد؟
Mastering scikit-learn 1.9 — many

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-21-ch21 (lesson 21)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch21.md
# ────────────────────────────────────────────────────────────────────────
import math
import numpy as np

dimensions = [2, 4, 8, 16, 24, 32]
print("نسبة حجم الكرة المحصورة (نصف القطر 0.5) داخل مكعب الوحدة:")

for d in dimensions:
    # حجم الكرة الفائقة في البعد d
    sphere_volume = (np.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0) * (0.5 ** d)
    print(f"  البعد d={d:>2} -> نسبة الحجم: {sphere_volume:11.8f} (المكعب فارغ بنسبة {(1 - sphere_volume) * 100:.4f}%)")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch21.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

dimensions_to_test = [2, 10, 50, 200, 1000]
rng = np.random.default_rng(42)

print("مقارنة المسافة بين أقرب نقطة وأبعد نقطة لمئة عينة عشوائية:")
for d in dimensions_to_test:
    # توليد 100 عينة عشوائية في فضاء ذي d بعداً
    points = rng.uniform(0.0, 1.0, size=(100, d))

    # حساب المسافات الإقليدية الزوجية بين جميع النقاط
    diffs = points[:, None, :] - points[None, :, :]
    pairwise_distances = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # استبعاد المسافة الصفرية بين النقطة ونفسها
    np.fill_diagonal(pairwise_distances, np.nan)

    min_dist = np.nanmin(pairwise_distances)
    max_dist = np.nanmax(pairwise_distances)
    relative_contrast = (max_dist - min_dist) / min_dist

    print(f"  البعد d={d:>4} | أقرب مسافة={min_dist:5.2f} | أبعد مسافة={max_dist:5.2f} | التباين النسبي={relative_contrast:6.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch21.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1. توليد بيانات نقية: 5 ميزات دالة فقط
X_pure, y_task = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    random_state=42
)

# قياس أداء k-NN على البيانات النقية
clean_knn_score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_pure, y_task, cv=5).mean()

# 2. حقن 100 ميزة ضجيج عشوائي لا علاقة لها بالهدف
rng_noise = np.random.default_rng(42)
pure_noise = rng_noise.normal(0, 1, size=(500, 100))
X_contaminated = np.column_stack([X_pure, pure_noise])

# قياس أداء k-NN بعد التلويث بالضجيج
noisy_knn_score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_contaminated, y_task, cv=5).mean()

# 3. اختبار نموذج خطي منظّم عبر L1 (Lasso Penalty) في 1.9
# في الإصدارات الحديثة نضبط l1_ratio=1.0 مع solver='saga'
l1_model = LogisticRegression(
    l1_ratio=1.0,
    solver="saga",
    C=0.2,
    max_iter=500,
    random_state=42
)
noisy_l1_score = cross_val_score(l1_model, X_contaminated, y_task, cv=5).mean()

print(f"دقة k-NN على الميزات الأصلية النقية (5 ميزات)        : {clean_knn_score:.1%}")
print(f"دقة k-NN بعد إضافة 100 ميزة ضجيج عشوائي             : {noisy_knn_score:.1%}")
print(f"نسبة الانهيار في دقة الجيران بسبب لعنة الأبعاد      : -{(clean_knn_score - noisy_knn_score) * 100:.1f}%\n")
print(f"دقة النموذج المنظم (L1 Regularization) على نفس الضجيج: {noisy_l1_score:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch21.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# بناء خط أنابيب يدمج ضغط الأبعاد مع مصنّف الجيران
pca_knn_pipe = make_pipeline(
    PCA(n_components=5, random_state=42),
    KNeighborsClassifier(n_neighbors=5)
)
recovered_knn_score = cross_val_score(pca_knn_pipe, X_contaminated, y_task, cv=5).mean()

print("نتائج معالجة لعنة الأبعاد بواسطة تقليل الأبعاد PCA:")
print(f"  - دقة k-NN على 105 ميزات ملوثة         : {noisy_knn_score:.1%}")
print(f"  - دقة k-NN بعد ضغطها إلى 5 مكونات عبر PCA: {recovered_knn_score:.1%}")
print(f"  - نسبة التعافي في الأداء بفضل تقليل الأبعاد: +{(recovered_knn_score - noisy_knn_score) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-21-ch21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import math
import numpy as np

dimensions = [2, 4, 8, 16, 24, 32]
print("نسبة حجم الكرة المحصورة (نصف القطر 0.5) داخل مكعب الوحدة:")

for d in dimensions:
    # حجم الكرة الفائقة في البعد d
    sphere_volume = (np.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0) * (0.5 ** d)
    print(f"  البعد d={d:>2} -> نسبة الحجم: {sphere_volume:11.8f} (المكعب فارغ بنسبة {(1 - sphere_volume) * 100:.4f}%)")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-21-ch21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

dimensions_to_test = [2, 10, 50, 200, 1000]
rng = np.random.default_rng(42)

print("مقارنة المسافة بين أقرب نقطة وأبعد نقطة لمئة عينة عشوائية:")
for d in dimensions_to_test:
    # توليد 100 عينة عشوائية في فضاء ذي d بعداً
    points = rng.uniform(0.0, 1.0, size=(100, d))

    # حساب المسافات الإقليدية الزوجية بين جميع النقاط
    diffs = points[:, None, :] - points[None, :, :]
    pairwise_distances = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # استبعاد المسافة الصفرية بين النقطة ونفسها
    np.fill_diagonal(pairwise_distances, np.nan)

    min_dist = np.nanmin(pairwise_distances)
    max_dist = np.nanmax(pairwise_distances)
    relative_contrast = (max_dist - min_dist) / min_dist

    print(f"  البعد d={d:>4} | أقرب مسافة={min_dist:5.2f} | أبعد مسافة={max_dist:5.2f} | التباين النسبي={relative_contrast:6.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-21-ch21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1. توليد بيانات نقية: 5 ميزات دالة فقط
X_pure, y_task = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    random_state=42
)

# قياس أداء k-NN على البيانات النقية
clean_knn_score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_pure, y_task, cv=5).mean()

# 2. حقن 100 ميزة ضجيج عشوائي لا علاقة لها بالهدف
rng_noise = np.random.default_rng(42)
pure_noise = rng_noise.normal(0, 1, size=(500, 100))
X_contaminated = np.column_stack([X_pure, pure_noise])

# قياس أداء k-NN بعد التلويث بالضجيج
noisy_knn_score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_contaminated, y_task, cv=5).mean()

# 3. اختبار نموذج خطي منظّم عبر L1 (Lasso Penalty) في 1.9
# في الإصدارات الحديثة نضبط l1_ratio=1.0 مع solver='saga'
l1_model = LogisticRegression(
    l1_ratio=1.0,
    solver="saga",
    C=0.2,
    max_iter=500,
    random_state=42
)
noisy_l1_score = cross_val_score(l1_model, X_contaminated, y_task, cv=5).mean()

print(f"دقة k-NN على الميزات الأصلية النقية (5 ميزات)        : {clean_knn_score:.1%}")
print(f"دقة k-NN بعد إضافة 100 ميزة ضجيج عشوائي             : {noisy_knn_score:.1%}")
print(f"نسبة الانهيار في دقة الجيران بسبب لعنة الأبعاد      : -{(clean_knn_score - noisy_knn_score) * 100:.1f}%\n")
print(f"دقة النموذج المنظم (L1 Regularization) على نفس الضجيج: {noisy_l1_score:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-21-ch21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# بناء خط أنابيب يدمج ضغط الأبعاد مع مصنّف الجيران
pca_knn_pipe = make_pipeline(
    PCA(n_components=5, random_state=42),
    KNeighborsClassifier(n_neighbors=5)
)
recovered_knn_score = cross_val_score(pca_knn_pipe, X_contaminated, y_task, cv=5).mean()

print("نتائج معالجة لعنة الأبعاد بواسطة تقليل الأبعاد PCA:")
print(f"  - دقة k-NN على 105 ميزات ملوثة         : {noisy_knn_score:.1%}")
print(f"  - دقة k-NN بعد ضغطها إلى 5 مكونات عبر PCA: {recovered_knn_score:.1%}")
print(f"  - نسبة التعافي في الأداء بفضل تقليل الأبعاد: +{(recovered_knn_score - noisy_knn_score) * 100:.1f}%")
