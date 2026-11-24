#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصفية التي لا تنظر إلى النموذج
Mastering scikit-learn 1.9 — Chapter 22 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-22-ch22 (lesson 22)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-22-ch22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصفية التي لا تنظر إلى النموذج
Mastering scikit-learn 1.9 — Chapter 22 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-22-ch22 (lesson 22)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch22.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# مصفوفة بيانات تجريبية:
# العمود 0: ثابت تماماً (التباين = 0)
# العمود 1: شبه ثابت (قيمة 0 في 80% من الحالات)
# العمود 2 و 3: ميزات طبيعية التباين
X_raw = np.array([
    [1.0, 0.0, 3.4, 120.0],
    [1.0, 0.0, 1.2, 150.0],
    [1.0, 0.0, 5.1, 110.0],
    [1.0, 1.0, 2.8, 140.0],
    [1.0, 0.0, 4.0, 130.0]
])

# استبعاد الميزات التي يقل تباينها عن عتبة محددة (0.15)
selector = VarianceThreshold(threshold=0.15)
X_clean = selector.fit_transform(X_raw)

print(f"أبعاد المصفوفة الأصلية: {X_raw.shape} (4 ميزات)")
print(f"أبعاد المصفوفة المصفاة : {X_clean.shape} (ميزتان فقط)")
print("قناع الميزات المحتفظ بها (get_support):", selector.get_support())
print("تباين كل عمود محسوباً بدقة       :", selector.variances_.round(3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch22.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

X_iris, y_iris = load_iris(return_X_y=True)

# اختيار أفضل ميزتين باستخدام اختبار التباين ANOVA (f_classif)
kbest_selector = SelectKBest(score_func=f_classif, k=2)
X_iris_selected = kbest_selector.fit_transform(X_iris, y_iris)

scores = kbest_selector.scores_
p_values = kbest_selector.pvalues_

print("أبعاد Iris الأصلية :", X_iris.shape)
print("أبعاد بعد الانتقاء  :", X_iris_selected.shape)
print("\nدرجات F الإحصائية لكل ميزة من الميزات الأربع:")
for i, (sc, pv) in enumerate(zip(scores, p_values)):
    print(f"  الميزة {i}: درجة F = {sc:7.1f} | القيمة الاحتمالية p-value = {pv:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch22.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_selection import f_classif, mutual_info_classif

rng = np.random.default_rng(42)
n_samples = 600

# 1. ميزة مرتبطة بالهدف بعلاقة تربيعية متماثلة حول الصفر (العلاقة الخطية معدومة)
x_parabola = rng.uniform(-2.5, 2.5, n_samples)
y_binary = (x_parabola ** 2 > 1.5).astype(int)

# 2. ميزة ضجيج عشوائية مستقلة تماماً
x_pure_noise = rng.normal(0, 1, n_samples)

X_comparison = np.column_stack([x_parabola, x_pure_noise])

# حساب F-test
f_scores, f_pvals = f_classif(X_comparison, y_binary)
# حساب المعلومات المتبادلة MI
mi_scores = mutual_info_classif(X_comparison, y_binary, random_state=42)

print("مقارنة قياس العلاقة التربيعية بين المقياسين:")
print(f"  الميزة غير الخطية (x^2) : F-score = {f_scores[0]:5.2f} (p={f_pvals[0]:.2f}) | MI = {mi_scores[0]:.3f}")
print(f"  ميزة الضجيج العشوائي     : F-score = {f_scores[1]:5.2f} (p={f_pvals[1]:.2f}) | MI = {mi_scores[1]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch22.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# توليد بيانات ذات 50 ميزة (10 ميزات حقيقية و 40 ميزة ضجيج)
X_large, y_large = make_classification(
    n_samples=500,
    n_features=50,
    n_informative=10,
    n_redundant=0,
    random_state=42
)

# بناء خط أنابيب آمن يختار أفضل 20% فقط من الميزات (أي 10 ميزات)
secure_filter_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    LogisticRegression(max_iter=300, random_state=42)
)

# تقييم خط الأنابيب بالتحقق المتقاطع (التصفية تُحسب داخل كل طية تدريب فقط)
cv_filtered_accuracy = cross_val_score(secure_filter_pipeline, X_large, y_large, cv=5).mean()

# مقارنة مع النموذج الذي يبتلع كافة الميزات الخمسين بلا تصفية
baseline_full_accuracy = cross_val_score(LogisticRegression(max_iter=300, random_state=42), X_large, y_large, cv=5).mean()

print(f"دقة النموذج مع كافة الميزات (50 ميزة)           : {baseline_full_accuracy:.1%}")
print(f"دقة النموذج بعد تصفية أفضل 20% فقط (10 ميزات دالة): {cv_filtered_accuracy:.1%}")
print(f"التحسن في الدقة مع سرعة استدلال مضاعفة خمس مرات: +{(cv_filtered_accuracy - baseline_full_accuracy) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-22-ch22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# مصفوفة بيانات تجريبية:
# العمود 0: ثابت تماماً (التباين = 0)
# العمود 1: شبه ثابت (قيمة 0 في 80% من الحالات)
# العمود 2 و 3: ميزات طبيعية التباين
X_raw = np.array([
    [1.0, 0.0, 3.4, 120.0],
    [1.0, 0.0, 1.2, 150.0],
    [1.0, 0.0, 5.1, 110.0],
    [1.0, 1.0, 2.8, 140.0],
    [1.0, 0.0, 4.0, 130.0]
])

# استبعاد الميزات التي يقل تباينها عن عتبة محددة (0.15)
selector = VarianceThreshold(threshold=0.15)
X_clean = selector.fit_transform(X_raw)

print(f"أبعاد المصفوفة الأصلية: {X_raw.shape} (4 ميزات)")
print(f"أبعاد المصفوفة المصفاة : {X_clean.shape} (ميزتان فقط)")
print("قناع الميزات المحتفظ بها (get_support):", selector.get_support())
print("تباين كل عمود محسوباً بدقة       :", selector.variances_.round(3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-22-ch22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

X_iris, y_iris = load_iris(return_X_y=True)

# اختيار أفضل ميزتين باستخدام اختبار التباين ANOVA (f_classif)
kbest_selector = SelectKBest(score_func=f_classif, k=2)
X_iris_selected = kbest_selector.fit_transform(X_iris, y_iris)

scores = kbest_selector.scores_
p_values = kbest_selector.pvalues_

print("أبعاد Iris الأصلية :", X_iris.shape)
print("أبعاد بعد الانتقاء  :", X_iris_selected.shape)
print("\nدرجات F الإحصائية لكل ميزة من الميزات الأربع:")
for i, (sc, pv) in enumerate(zip(scores, p_values)):
    print(f"  الميزة {i}: درجة F = {sc:7.1f} | القيمة الاحتمالية p-value = {pv:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-22-ch22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_selection import f_classif, mutual_info_classif

rng = np.random.default_rng(42)
n_samples = 600

# 1. ميزة مرتبطة بالهدف بعلاقة تربيعية متماثلة حول الصفر (العلاقة الخطية معدومة)
x_parabola = rng.uniform(-2.5, 2.5, n_samples)
y_binary = (x_parabola ** 2 > 1.5).astype(int)

# 2. ميزة ضجيج عشوائية مستقلة تماماً
x_pure_noise = rng.normal(0, 1, n_samples)

X_comparison = np.column_stack([x_parabola, x_pure_noise])

# حساب F-test
f_scores, f_pvals = f_classif(X_comparison, y_binary)
# حساب المعلومات المتبادلة MI
mi_scores = mutual_info_classif(X_comparison, y_binary, random_state=42)

print("مقارنة قياس العلاقة التربيعية بين المقياسين:")
print(f"  الميزة غير الخطية (x^2) : F-score = {f_scores[0]:5.2f} (p={f_pvals[0]:.2f}) | MI = {mi_scores[0]:.3f}")
print(f"  ميزة الضجيج العشوائي     : F-score = {f_scores[1]:5.2f} (p={f_pvals[1]:.2f}) | MI = {mi_scores[1]:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-22-ch22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# توليد بيانات ذات 50 ميزة (10 ميزات حقيقية و 40 ميزة ضجيج)
X_large, y_large = make_classification(
    n_samples=500,
    n_features=50,
    n_informative=10,
    n_redundant=0,
    random_state=42
)

# بناء خط أنابيب آمن يختار أفضل 20% فقط من الميزات (أي 10 ميزات)
secure_filter_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    LogisticRegression(max_iter=300, random_state=42)
)

# تقييم خط الأنابيب بالتحقق المتقاطع (التصفية تُحسب داخل كل طية تدريب فقط)
cv_filtered_accuracy = cross_val_score(secure_filter_pipeline, X_large, y_large, cv=5).mean()

# مقارنة مع النموذج الذي يبتلع كافة الميزات الخمسين بلا تصفية
baseline_full_accuracy = cross_val_score(LogisticRegression(max_iter=300, random_state=42), X_large, y_large, cv=5).mean()

print(f"دقة النموذج مع كافة الميزات (50 ميزة)           : {baseline_full_accuracy:.1%}")
print(f"دقة النموذج بعد تصفية أفضل 20% فقط (10 ميزات دالة): {cv_filtered_accuracy:.1%}")
print(f"التحسن في الدقة مع سرعة استدلال مضاعفة خمس مرات: +{(cv_filtered_accuracy - baseline_full_accuracy) * 100:.1f}%")
