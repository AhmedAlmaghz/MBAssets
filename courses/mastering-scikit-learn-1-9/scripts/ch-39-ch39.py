#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التشذيب والإفراط في الملاءمة
Mastering scikit-learn 1.9 — Chapter 39 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-39-ch39 (lesson 39)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-39-ch39.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التشذيب والإفراط في الملاءمة
Mastering scikit-learn 1.9 — Chapter 39 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-39-ch39 (lesson 39)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part05/ch39.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# تدريب شجرة كاملة أولاً
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(f"عدد قيم alpha الحرجة المستخرجة: {len(ccp_alphas)}")
print(f"أصغر قيمة alpha (الشجرة الكاملة): {ccp_alphas[0]:.6f}")
print(f"أكبر قيمة alpha (تقليص لجذر وحيد): {ccp_alphas[-1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part05/ch39.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
# استبعاد القيمة الأخيرة لأنها تشذب كل شيء وتترك ورقة واحدة
ccp_alphas = path.ccp_alphas[:-1]

# فحص عينة من قيم alpha بالتحقق المتقاطع
selected_alphas = ccp_alphas[::max(1, len(ccp_alphas)//10)]
cv_scores = []

for alpha in selected_alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(t, X_train, y_train, cv=5).mean()
    cv_scores.append(score)

best_idx = np.argmax(cv_scores)
best_alpha = selected_alphas[best_idx]

# تقييم الشجرة المشذبة مقابل الشجرة غير المشذبة على الاختبار المستقل
tree_unpruned = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
tree_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42).fit(X_train, y_train)

print(f"أفضل ccp_alpha مختار: {best_alpha:.5f}")
print(f"الشجرة غير المشذبة | عدد الأوراق: {tree_unpruned.get_n_leaves():2d} | دقة الاختبار: {tree_unpruned.score(X_test, y_test):.4f}")
print(f"الشجرة المشذبة    | عدد الأوراق: {tree_pruned.get_n_leaves():2d} | دقة الاختبار: {tree_pruned.score(X_test, y_test):.4f}")
print(f"انخفاض حجم الشجرة بنسبة: {1 - tree_pruned.get_n_leaves()/tree_unpruned.get_n_leaves():.1%} مع تحسن التعميم!")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part05/ch39.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

# استخراج مسار القرار للعينة الأولى
sample = X[:1]
node_indicator = clf.decision_path(sample)
leaf_id = clf.apply(sample)

node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

print(f"العينة تنتمي في النهاية للورقة رقم: {leaf_id[0]}")
print("تسلسل العقد التي مرت بها العينة في رحلة اتخاذ القرار:")
for node_id in node_index:
    if leaf_id[0] == node_id:
        print(f"  --> العقدة الطرفية #{node_id} (إصدار الحكم)")
    else:
        feat = clf.tree_.feature[node_id]
        thresh = clf.tree_.threshold[node_id]
        print(f"  العقدة #{node_id}: فحص الميزة '{feat}' بمقارنتها مع العتبة {thresh:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part05/ch39.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 1. تقييد قبلي
pre_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)

# 2. تشذيب بعدي
full_tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
# اختيار alpha متوسط فعال
post_tree = DecisionTreeClassifier(ccp_alpha=path.ccp_alphas[len(path.ccp_alphas)//2], random_state=42).fit(X_tr, y_tr)

print(f"دقة التقييد القبلي (max_depth=3): {pre_tree.score(X_te, y_te):.4f} (أوراق: {pre_tree.get_n_leaves()})")
print(f"دقة التشذيب البعدي (ccp_alpha):   {post_tree.score(X_te, y_te):.4f} (أوراق: {post_tree.get_n_leaves()})")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part05/ch39.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42).fit(X, y)

# فحص مسار 15 عينة
indicators = clf.decision_path(X[:15])
# عدد العقد المزارة هو عدد الواحدات في كل صف
path_lengths = np.diff(indicators.indptr)

print(f"أطوال مسارات اتخاذ القرار للعينات: {list(path_lengths)}")
print(f"متوسط عدد الأسئلة اللازمة لإصدار الحكم: {path_lengths.mean():.1f} سؤالاً")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-39-ch39.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# تدريب شجرة كاملة أولاً
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(f"عدد قيم alpha الحرجة المستخرجة: {len(ccp_alphas)}")
print(f"أصغر قيمة alpha (الشجرة الكاملة): {ccp_alphas[0]:.6f}")
print(f"أكبر قيمة alpha (تقليص لجذر وحيد): {ccp_alphas[-1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-39-ch39.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
# استبعاد القيمة الأخيرة لأنها تشذب كل شيء وتترك ورقة واحدة
ccp_alphas = path.ccp_alphas[:-1]

# فحص عينة من قيم alpha بالتحقق المتقاطع
selected_alphas = ccp_alphas[::max(1, len(ccp_alphas)//10)]
cv_scores = []

for alpha in selected_alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(t, X_train, y_train, cv=5).mean()
    cv_scores.append(score)

best_idx = np.argmax(cv_scores)
best_alpha = selected_alphas[best_idx]

# تقييم الشجرة المشذبة مقابل الشجرة غير المشذبة على الاختبار المستقل
tree_unpruned = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
tree_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42).fit(X_train, y_train)

print(f"أفضل ccp_alpha مختار: {best_alpha:.5f}")
print(f"الشجرة غير المشذبة | عدد الأوراق: {tree_unpruned.get_n_leaves():2d} | دقة الاختبار: {tree_unpruned.score(X_test, y_test):.4f}")
print(f"الشجرة المشذبة    | عدد الأوراق: {tree_pruned.get_n_leaves():2d} | دقة الاختبار: {tree_pruned.score(X_test, y_test):.4f}")
print(f"انخفاض حجم الشجرة بنسبة: {1 - tree_pruned.get_n_leaves()/tree_unpruned.get_n_leaves():.1%} مع تحسن التعميم!")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-39-ch39.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

# استخراج مسار القرار للعينة الأولى
sample = X[:1]
node_indicator = clf.decision_path(sample)
leaf_id = clf.apply(sample)

node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

print(f"العينة تنتمي في النهاية للورقة رقم: {leaf_id[0]}")
print("تسلسل العقد التي مرت بها العينة في رحلة اتخاذ القرار:")
for node_id in node_index:
    if leaf_id[0] == node_id:
        print(f"  --> العقدة الطرفية #{node_id} (إصدار الحكم)")
    else:
        feat = clf.tree_.feature[node_id]
        thresh = clf.tree_.threshold[node_id]
        print(f"  العقدة #{node_id}: فحص الميزة '{feat}' بمقارنتها مع العتبة {thresh:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-39-ch39.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 1. تقييد قبلي
pre_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)

# 2. تشذيب بعدي
full_tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
# اختيار alpha متوسط فعال
post_tree = DecisionTreeClassifier(ccp_alpha=path.ccp_alphas[len(path.ccp_alphas)//2], random_state=42).fit(X_tr, y_tr)

print(f"دقة التقييد القبلي (max_depth=3): {pre_tree.score(X_te, y_te):.4f} (أوراق: {pre_tree.get_n_leaves()})")
print(f"دقة التشذيب البعدي (ccp_alpha):   {post_tree.score(X_te, y_te):.4f} (أوراق: {post_tree.get_n_leaves()})")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-39-ch39.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42).fit(X, y)

# فحص مسار 15 عينة
indicators = clf.decision_path(X[:15])
# عدد العقد المزارة هو عدد الواحدات في كل صف
path_lengths = np.diff(indicators.indptr)

print(f"أطوال مسارات اتخاذ القرار للعينات: {list(path_lengths)}")
print(f"متوسط عدد الأسئلة اللازمة لإصدار الحكم: {path_lengths.mean():.1f} سؤالاً")
