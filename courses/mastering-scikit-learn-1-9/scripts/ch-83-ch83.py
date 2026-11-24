#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — اختبار الأهمية الإحصائية
Mastering scikit-learn 1.9 — Chapter 83 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-83-ch83 (lesson 83)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-83-ch83.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — اختبار الأهمية الإحصائية
Mastering scikit-learn 1.9 — Chapter 83 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-83-ch83 (lesson 83)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2))

means = []
for seed in range(8):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    means.append(cross_val_score(pipe, X, y, cv=cv, n_jobs=-1).mean())

print("متوسطات البذور الثماني:", np.round(means, 4))
print(f"المدى الكامل للتذبذب: {max(means) - min(means):.4f}")
print(f"المتوسط العام: {np.mean(means):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import permutation_test_score

score, perm_scores, pvalue = permutation_test_score(
    pipe, X, y, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {score:.4f} | p-value: {pvalue:.4f}")
print(f"توزيع الصفر — المتوسط: {perm_scores.mean():.4f} | الأعلى: {perm_scores.max():.4f}")
print(f"عدد التبديلات التي تفوقت على الأصل: {(perm_scores >= score).sum()} من 100")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X_amb, y_amb = make_classification(n_samples=300, n_features=6, n_informative=2,
                                   n_redundant=1, class_sep=0.05, flip_y=0.30,
                                   random_state=42)
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

amb_score, amb_perm, amb_p = permutation_test_score(
    pipe_lr, X_amb, y_amb, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {amb_score:.4f} | متوسط توزيع الصفر: {amb_perm.mean():.4f}")
print(f"p-value: {amb_p:.4f}")
print(f"هل تقف الدرجة داخل توزيع الصفر؟ {(amb_score < amb_perm.max()) and (amb_score > amb_perm.min())}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import permutation_test_score

for n_perm in (100, 500):
    t0 = time.perf_counter()
    _, _, p_n = permutation_test_score(
        pipe_lr, X_amb, y_amb, scoring="accuracy",
        n_permutations=n_perm, random_state=42, n_jobs=-1)
    t_elapsed = time.perf_counter() - t0
    print(f"n_permutations={n_perm:3d} | p-value={p_n:.4f} | أصغر p ممكن: {1/(n_perm+1):.4f} | الزمن: {t_elapsed:.1f} s")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import ttest_rel

cv = StratifiedKFold(5, shuffle=True, random_state=42)   # نفس الطيات للاثنين حرفياً
m1 = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2))
m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

s1 = cross_val_score(m1, X, y, cv=cv, n_jobs=-1)
s2 = cross_val_score(m2, X, y, cv=cv, n_jobs=-1)
d = s1 - s2
t_stat, p_val = ttest_rel(s1, s2)

n = len(y)
n_train, n_test = int(0.8 * n), n - int(0.8 * n)
t_nb = d.mean() / np.sqrt(np.var(d, ddof=1) * (1 / 5 + n_test / n_train))

print("طيات SVC:            ", np.round(s1, 4))
print("طيات LogisticRegression:", np.round(s2, 4))
print(f"فروق الطيات: {np.round(d, 4)} | متوسط الفرق: {d.mean():+.4f}")
print(f"اختبار تي المزدوج: t={t_stat:.3f}, p={p_val:.4f}")
print(f"تي المصحح (Nadeau-Bengio): t={t_nb:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import permutation_test_score

Xw, yw = load_wine(return_X_y=True)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

w_score, w_perm, w_p = permutation_test_score(
    tree, Xw, yw, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {w_score:.4f} | p-value: {w_p:.4f}")
print(f"متوسط توزيع الصفر: {w_perm.mean():.4f}")
print(f"أصغر p ممكن مع 100 تبديل: {1/101:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch83.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel

Xd, yd = load_digits(return_X_y=True)
Xd, yd = Xd[:600], yd[:600]
cv_d = StratifiedKFold(5, shuffle=True, random_state=42)
lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
forrest = make_pipeline(StandardScaler(),
                        RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))

a = cross_val_score(lin, Xd, yd, cv=cv_d, n_jobs=-1)
b = cross_val_score(forrest, Xd, yd, cv=cv_d, n_jobs=-1)
t_st, p_st = ttest_rel(a, b)
n_d = len(yd)
t_nb_d = (a - b).mean() / np.sqrt(
    np.var(a - b, ddof=1) * (1 / 5 + (n_d - int(0.8 * n_d)) / int(0.8 * n_d)))

print(f"لوجيستي: {np.round(a, 4)} | متوسط {a.mean():.4f}")
print(f"غابة:    {np.round(b, 4)} | متوسط {b.mean():.4f}")
print(f"t={t_st:.3f}, p={p_st:.4f} | t المصحح={t_nb_d:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2))

means = []
for seed in range(8):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    means.append(cross_val_score(pipe, X, y, cv=cv, n_jobs=-1).mean())

print("متوسطات البذور الثماني:", np.round(means, 4))
print(f"المدى الكامل للتذبذب: {max(means) - min(means):.4f}")
print(f"المتوسط العام: {np.mean(means):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import permutation_test_score

score, perm_scores, pvalue = permutation_test_score(
    pipe, X, y, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {score:.4f} | p-value: {pvalue:.4f}")
print(f"توزيع الصفر — المتوسط: {perm_scores.mean():.4f} | الأعلى: {perm_scores.max():.4f}")
print(f"عدد التبديلات التي تفوقت على الأصل: {(perm_scores >= score).sum()} من 100")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X_amb, y_amb = make_classification(n_samples=300, n_features=6, n_informative=2,
                                   n_redundant=1, class_sep=0.05, flip_y=0.30,
                                   random_state=42)
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

amb_score, amb_perm, amb_p = permutation_test_score(
    pipe_lr, X_amb, y_amb, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {amb_score:.4f} | متوسط توزيع الصفر: {amb_perm.mean():.4f}")
print(f"p-value: {amb_p:.4f}")
print(f"هل تقف الدرجة داخل توزيع الصفر؟ {(amb_score < amb_perm.max()) and (amb_score > amb_perm.min())}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import permutation_test_score

for n_perm in (100, 500):
    t0 = time.perf_counter()
    _, _, p_n = permutation_test_score(
        pipe_lr, X_amb, y_amb, scoring="accuracy",
        n_permutations=n_perm, random_state=42, n_jobs=-1)
    t_elapsed = time.perf_counter() - t0
    print(f"n_permutations={n_perm:3d} | p-value={p_n:.4f} | أصغر p ممكن: {1/(n_perm+1):.4f} | الزمن: {t_elapsed:.1f} s")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from scipy.stats import ttest_rel

cv = StratifiedKFold(5, shuffle=True, random_state=42)   # نفس الطيات للاثنين حرفياً
m1 = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2))
m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

s1 = cross_val_score(m1, X, y, cv=cv, n_jobs=-1)
s2 = cross_val_score(m2, X, y, cv=cv, n_jobs=-1)
d = s1 - s2
t_stat, p_val = ttest_rel(s1, s2)

n = len(y)
n_train, n_test = int(0.8 * n), n - int(0.8 * n)
t_nb = d.mean() / np.sqrt(np.var(d, ddof=1) * (1 / 5 + n_test / n_train))

print("طيات SVC:            ", np.round(s1, 4))
print("طيات LogisticRegression:", np.round(s2, 4))
print(f"فروق الطيات: {np.round(d, 4)} | متوسط الفرق: {d.mean():+.4f}")
print(f"اختبار تي المزدوج: t={t_stat:.3f}, p={p_val:.4f}")
print(f"تي المصحح (Nadeau-Bengio): t={t_nb:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import permutation_test_score

Xw, yw = load_wine(return_X_y=True)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

w_score, w_perm, w_p = permutation_test_score(
    tree, Xw, yw, scoring="accuracy",
    n_permutations=100, random_state=42, n_jobs=-1)

print(f"الدرجة الأصلية: {w_score:.4f} | p-value: {w_p:.4f}")
print(f"متوسط توزيع الصفر: {w_perm.mean():.4f}")
print(f"أصغر p ممكن مع 100 تبديل: {1/101:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-83-ch83.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel

Xd, yd = load_digits(return_X_y=True)
Xd, yd = Xd[:600], yd[:600]
cv_d = StratifiedKFold(5, shuffle=True, random_state=42)
lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
forrest = make_pipeline(StandardScaler(),
                        RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))

a = cross_val_score(lin, Xd, yd, cv=cv_d, n_jobs=-1)
b = cross_val_score(forrest, Xd, yd, cv=cv_d, n_jobs=-1)
t_st, p_st = ttest_rel(a, b)
n_d = len(yd)
t_nb_d = (a - b).mean() / np.sqrt(
    np.var(a - b, ddof=1) * (1 / 5 + (n_d - int(0.8 * n_d)) / int(0.8 * n_d)))

print(f"لوجيستي: {np.round(a, 4)} | متوسط {a.mean():.4f}")
print(f"غابة:    {np.round(b, 4)} | متوسط {b.mean():.4f}")
print(f"t={t_st:.3f}, p={p_st:.4f} | t المصحح={t_nb_d:.3f}")
