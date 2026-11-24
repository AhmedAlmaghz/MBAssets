#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قراءة الشجرة وتفسيرها
Mastering scikit-learn 1.9 — Chapter 41 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-41-ch41 (lesson 41)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-41-ch41.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قراءة الشجرة وتفسيرها
Mastering scikit-learn 1.9 — Chapter 41 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-41-ch41 (lesson 41)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

# تصدير منطق الشجرة بأسماء الميزات والفئات الصريحة
tree_rules = export_text(
    clf,
    feature_names=list(iris.feature_names),
    class_names=list(iris.target_names),
    spacing=3,
    decimals=2
)

print("القواعد المنطقية الصريحة المستخرجة من الشجرة:")
print(tree_rules[:450])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

fig, ax = plt.subplots(figsize=(8, 5))
_ = plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)
plt.tight_layout()
plt.close(fig)

print(f"تم بناء المخطط البصري للشجرة بنجاح بعمق {clf.get_depth()} وعدد أوراق {clf.get_n_leaves()}!")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

tree = clf.tree_
n_nodes = tree.node_count

# حساب الأهمية يدوياً
manual_importances = np.zeros(X.shape[1])
total_samples = tree.n_node_samples[0]

for node in range(n_nodes):
    # فحص إذا كانت العقدة غير طرفية (لها أبناء)
    if tree.children_left[node] != -1:
        feat = tree.feature[node]
        left = tree.children_left[node]
        right = tree.children_right[node]

        n_parent = tree.n_node_samples[node]
        n_left = tree.n_node_samples[left]
        n_right = tree.n_node_samples[right]

        imp_parent = tree.impurity[node]
        imp_left = tree.impurity[left]
        imp_right = tree.impurity[right]

        # انخفاض الشوائب الموزون
        decrease = (n_parent / total_samples) * imp_parent - \
                   (n_left / total_samples) * imp_left - \
                   (n_right / total_samples) * imp_right

        manual_importances[feat] += decrease

# التطبيع ليصبح المجموع 1.0
manual_importances /= manual_importances.sum()

print("مقارنة حساب أهمية الميزات:")
for i, name in enumerate(iris.feature_names):
    print(f"  {name:<25} | اليدوي: {manual_importances[i]:.4f} | المدمج: {clf.feature_importances_[i]:.4f}")

print(f"\nتطابق الحساب اليدوي مع مخرجات scikit-learn التامة: {np.allclose(manual_importances, clf.feature_importances_)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(42)
n_samples = 400

# ميزة 0: ثنائية مفيدة جداً ترتبط بالهدف
x_real = rng.binomial(1, 0.5, size=n_samples)
y = x_real.copy()

# ميزة 1: أرقام عشوائية مستمرة بالكامل (ضجيج عديم الفائدة ولكن بقيم فريدة هائلة!)
x_noise = rng.uniform(0, 100, size=n_samples)

X_bias = np.column_stack([x_real, x_noise])

clf = DecisionTreeClassifier(random_state=42).fit(X_bias, y)

# حساب أهمية الشوائب المدمجة
impurity_imp = clf.feature_importances_

# حساب أهمية الإبدال المستقلة permutation_importance
perm_imp = permutation_importance(clf, X_bias, y, random_state=42).importances_mean

print(f"{'نوع الميزة':<25} {'أهمية الشوائب (المدمجة)':>25} {'أهمية الإبدال (الحقيقية)':>25}")
print("-" * 75)
print(f"{'الميزة الحقيقية (ثنائية)':<25} {impurity_imp[0]:>25.4f} {perm_imp[0]:>25.4f}")
print(f"{'الضجيج العشوائي (كثير التعدد)':<25} {impurity_imp[1]:>25.4f} {perm_imp[1]:>25.4f}")
print("\nالخلاصة: أهمية الشوائب المدمجة تمنح الضجيج أهمية زائفة؛ بينما تكشف أهمية الإبدال الحقيقة!")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text

cancer = load_breast_cancer()
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(cancer.data, cancer.target)

rules = export_text(clf, feature_names=list(cancer.feature_names), decimals=3)
first_rule = rules.strip().split("\n")[0]
print(f"قاعدة عقدة الجذر الأولى: {first_rule}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part05/ch41.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(iris.data, iris.target)

def get_depth_manual(node=0):
    if clf.tree_.children_left[node] == -1:
        return 0
    left_d = get_depth_manual(clf.tree_.children_left[node])
    right_d = get_depth_manual(clf.tree_.children_right[node])
    return 1 + max(left_d, right_d)

print(f"العمق المحسوب يدوياً: {get_depth_manual(0)}")
print(f"العمق عبر دالة المقدر: {clf.get_depth()}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

# تصدير منطق الشجرة بأسماء الميزات والفئات الصريحة
tree_rules = export_text(
    clf,
    feature_names=list(iris.feature_names),
    class_names=list(iris.target_names),
    spacing=3,
    decimals=2
)

print("القواعد المنطقية الصريحة المستخرجة من الشجرة:")
print(tree_rules[:450])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

fig, ax = plt.subplots(figsize=(8, 5))
_ = plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)
plt.tight_layout()
plt.close(fig)

print(f"تم بناء المخطط البصري للشجرة بنجاح بعمق {clf.get_depth()} وعدد أوراق {clf.get_n_leaves()}!")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

tree = clf.tree_
n_nodes = tree.node_count

# حساب الأهمية يدوياً
manual_importances = np.zeros(X.shape[1])
total_samples = tree.n_node_samples[0]

for node in range(n_nodes):
    # فحص إذا كانت العقدة غير طرفية (لها أبناء)
    if tree.children_left[node] != -1:
        feat = tree.feature[node]
        left = tree.children_left[node]
        right = tree.children_right[node]

        n_parent = tree.n_node_samples[node]
        n_left = tree.n_node_samples[left]
        n_right = tree.n_node_samples[right]

        imp_parent = tree.impurity[node]
        imp_left = tree.impurity[left]
        imp_right = tree.impurity[right]

        # انخفاض الشوائب الموزون
        decrease = (n_parent / total_samples) * imp_parent - \
                   (n_left / total_samples) * imp_left - \
                   (n_right / total_samples) * imp_right

        manual_importances[feat] += decrease

# التطبيع ليصبح المجموع 1.0
manual_importances /= manual_importances.sum()

print("مقارنة حساب أهمية الميزات:")
for i, name in enumerate(iris.feature_names):
    print(f"  {name:<25} | اليدوي: {manual_importances[i]:.4f} | المدمج: {clf.feature_importances_[i]:.4f}")

print(f"\nتطابق الحساب اليدوي مع مخرجات scikit-learn التامة: {np.allclose(manual_importances, clf.feature_importances_)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(42)
n_samples = 400

# ميزة 0: ثنائية مفيدة جداً ترتبط بالهدف
x_real = rng.binomial(1, 0.5, size=n_samples)
y = x_real.copy()

# ميزة 1: أرقام عشوائية مستمرة بالكامل (ضجيج عديم الفائدة ولكن بقيم فريدة هائلة!)
x_noise = rng.uniform(0, 100, size=n_samples)

X_bias = np.column_stack([x_real, x_noise])

clf = DecisionTreeClassifier(random_state=42).fit(X_bias, y)

# حساب أهمية الشوائب المدمجة
impurity_imp = clf.feature_importances_

# حساب أهمية الإبدال المستقلة permutation_importance
perm_imp = permutation_importance(clf, X_bias, y, random_state=42).importances_mean

print(f"{'نوع الميزة':<25} {'أهمية الشوائب (المدمجة)':>25} {'أهمية الإبدال (الحقيقية)':>25}")
print("-" * 75)
print(f"{'الميزة الحقيقية (ثنائية)':<25} {impurity_imp[0]:>25.4f} {perm_imp[0]:>25.4f}")
print(f"{'الضجيج العشوائي (كثير التعدد)':<25} {impurity_imp[1]:>25.4f} {perm_imp[1]:>25.4f}")
print("\nالخلاصة: أهمية الشوائب المدمجة تمنح الضجيج أهمية زائفة؛ بينما تكشف أهمية الإبدال الحقيقة!")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text

cancer = load_breast_cancer()
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(cancer.data, cancer.target)

rules = export_text(clf, feature_names=list(cancer.feature_names), decimals=3)
first_rule = rules.strip().split("\n")[0]
print(f"قاعدة عقدة الجذر الأولى: {first_rule}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-41-ch41.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(iris.data, iris.target)

def get_depth_manual(node=0):
    if clf.tree_.children_left[node] == -1:
        return 0
    left_d = get_depth_manual(clf.tree_.children_left[node])
    right_d = get_depth_manual(clf.tree_.children_right[node])
    return 1 + max(left_d, right_d)

print(f"العمق المحسوب يدوياً: {get_depth_manual(0)}")
print(f"العمق عبر دالة المقدر: {clf.get_depth()}")
