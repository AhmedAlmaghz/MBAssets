#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — AdaBoost: التعلّم بالنقاط الصعبة
Mastering scikit-learn 1.9 — AdaBoost

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-45-ch45 (lesson 45)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-45-ch45.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — AdaBoost: التعلّم بالنقاط الصعبة
Mastering scikit-learn 1.9 — AdaBoost

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-45-ch45 (lesson 45)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

rng = np.random.default_rng(42)
n_experts, n_items, p_weak = 101, 1000, 0.55

# كل «متعلم ضعيف» يجيب صح باحتمال 55% على كل عينة
votes = rng.random((n_experts, n_items)) < p_weak
majority_correct = votes.mean(axis=0) > 0.5

print(f"دقة المتعلم الضعيف المفرد: {p_weak:.0%}")
print(f"دقة أغلبية 101 متعلم ضعيف مستقل: {majority_correct.mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
n, K = len(y), 3

# ---- الجولة الأولى يدوياً: أوزان متساوية ----
w1 = np.full(n, 1.0 / n)
stump1 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w1)
eps1 = np.average(stump1.predict(X) != y, weights=w1)
alpha1 = np.log((1 - eps1) / eps1) + np.log(K - 1)

# ---- تحديث الأوزان وإعادة التطبيع ----
w2 = w1 * np.exp(alpha1 * (stump1.predict(X) != y))
w2 = w2 / w2.sum()

# ---- الجولة الثانية يدوياً بالأوزان الجديدة ----
stump2 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w2)
eps2 = np.average(stump2.predict(X) != y, weights=w2)
alpha2 = np.log((1 - eps2) / eps2) + np.log(K - 1)

# ---- المقارنة مع scikit-learn ----
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=2, random_state=42,
).fit(X, y)

print(f"الجولة ١ | الخطأ الموزون: يدوياً {eps1:.6f} | المكتبة {ada.estimator_errors_[0]:.6f}")
print(f"الجولة ١ | معامل النموذج: يدوياً {alpha1:.6f} | المكتبة {ada.estimator_weights_[0]:.6f}")
print(f"الجولة ٢ | الخطأ الموزون: يدوياً {eps2:.6f} | المكتبة {ada.estimator_errors_[1]:.6f}")
print(f"الجولة ٢ | معامل النموذج: يدوياً {alpha2:.6f} | المكتبة {ada.estimator_weights_[1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

print("AdaBoostClassifier.get_params():", AdaBoostClassifier().get_params())

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=200, learning_rate=0.5, random_state=42,
).fit(X_tr, y_tr)

errs = ada.estimator_errors_
staged = list(ada.staged_score(X_te, y_te))
print(f"أخطاء الجذوع الأولى:  {np.round(errs[:5], 4)}")
print(f"أخطاء الجذوع الأخيرة: {np.round(errs[-3:], 4)}")
for k in [1, 10, 50, 100, 200]:
    print(f"بعد {k:3d} جولة | دقة الاختبار المتدرجة: {staged[k-1]:.4f}")
print(f"أفضل دقة متدرجة: {max(staged):.4f} عند الجولة رقم {int(np.argmax(staged)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1500], data.target[:1500], test_size=0.3, random_state=42
)

for loss in ["linear", "square", "exponential"]:
    reg = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=100, loss=loss, random_state=42,
    ).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, reg.predict(X_te))
    print(f"loss={loss:12s} | RMSE الاختبار: {rmse:.4f} | R²: {reg.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

for depth, name in [(1, "جذع (عمق 1)"), (3, "شجرة عمق 3"), (None, "شجرة كاملة")]:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth, random_state=42),
        n_estimators=50, random_state=42,
    ).fit(X_tr, y_tr)
    print(f"قاعدة: {name:12s} | تدريب: {ada.score(X_tr, y_tr):.4f} | اختبار: {ada.score(X_te, y_te):.4f}")

single = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"شجرة مفردة كاملة بلا تعزيز | تدريب: {single.score(X_tr, y_tr):.4f} | اختبار: {single.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=600, n_features=10, n_informative=4,
    flip_y=0.20, class_sep=0.8, random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=150, random_state=42,
).fit(X_tr, y_tr)

errs = ada.estimator_errors_
print(f"أخطاء الجذوع الأولى: {np.round(errs[:4], 4)}")
print(f"أخطاء آخر 4 جولات:   {np.round(errs[-4:], 4)}  (حد العشوائية = 0.5)")
staged_tr = list(ada.staged_score(X_tr, y_tr))
staged_te = list(ada.staged_score(X_te, y_te))
for k in [5, 25, 50, 100, 150]:
    print(f"بعد {k:3d} جولة | تدريب: {staged_tr[k-1]:.4f} | اختبار: {staged_te[k-1]:.4f}")
print(f"أفضل دقة اختبار: {max(staged_te):.4f} عند الجولة {int(np.argmax(staged_te)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=300, learning_rate=0.3, random_state=42,
).fit(X_tr, y_tr)

staged = list(ada.staged_score(X_te, y_te))
for k in [1, 10, 100]:
    print(f"الجولة {k:3d} | دقة الاختبار المتدرجة: {staged[k-1]:.4f}")
best_k = int(np.argmax(staged)) + 1
print(f"الجولة المثلى: {best_k} | دقتها: {staged[best_k-1]:.4f}")
print(f"النموذج الكامل (300 جولة): {ada.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part06/ch45.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
n, K = len(y), 3

w = np.full(n, 1.0 / n)
manual_eps, manual_alpha = [], []
for t in range(5):
    stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w)
    wrong = stump.predict(X) != y
    eps = np.average(wrong, weights=w)
    alpha = np.log((1 - eps) / eps) + np.log(K - 1)
    manual_eps.append(eps)
    manual_alpha.append(alpha)
    w = w * np.exp(alpha * wrong)
    w = w / w.sum()

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=5, random_state=42,
).fit(X, y)

print("الجولة | الخطأ يدوياً | الخطأ مكتبة | α يدوياً | α مكتبة")
for t in range(5):
    print(f"  {t+1}   |  {manual_eps[t]:.6f}  |  {ada.estimator_errors_[t]:.6f}  |"
          f" {manual_alpha[t]:.6f} | {ada.estimator_weights_[t]:.6f}")
diff_eps = np.max(np.abs(np.array(manual_eps) - ada.estimator_errors_))
diff_alpha = np.max(np.abs(np.array(manual_alpha) - ada.estimator_weights_))
print(f"أكبر فرق في الأخطاء: {diff_eps:.2e} | أكبر فرق في المعاملات: {diff_alpha:.2e}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

rng = np.random.default_rng(42)
n_experts, n_items, p_weak = 101, 1000, 0.55

# كل «متعلم ضعيف» يجيب صح باحتمال 55% على كل عينة
votes = rng.random((n_experts, n_items)) < p_weak
majority_correct = votes.mean(axis=0) > 0.5

print(f"دقة المتعلم الضعيف المفرد: {p_weak:.0%}")
print(f"دقة أغلبية 101 متعلم ضعيف مستقل: {majority_correct.mean():.1%}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
n, K = len(y), 3

# ---- الجولة الأولى يدوياً: أوزان متساوية ----
w1 = np.full(n, 1.0 / n)
stump1 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w1)
eps1 = np.average(stump1.predict(X) != y, weights=w1)
alpha1 = np.log((1 - eps1) / eps1) + np.log(K - 1)

# ---- تحديث الأوزان وإعادة التطبيع ----
w2 = w1 * np.exp(alpha1 * (stump1.predict(X) != y))
w2 = w2 / w2.sum()

# ---- الجولة الثانية يدوياً بالأوزان الجديدة ----
stump2 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w2)
eps2 = np.average(stump2.predict(X) != y, weights=w2)
alpha2 = np.log((1 - eps2) / eps2) + np.log(K - 1)

# ---- المقارنة مع scikit-learn ----
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=2, random_state=42,
).fit(X, y)

print(f"الجولة ١ | الخطأ الموزون: يدوياً {eps1:.6f} | المكتبة {ada.estimator_errors_[0]:.6f}")
print(f"الجولة ١ | معامل النموذج: يدوياً {alpha1:.6f} | المكتبة {ada.estimator_weights_[0]:.6f}")
print(f"الجولة ٢ | الخطأ الموزون: يدوياً {eps2:.6f} | المكتبة {ada.estimator_errors_[1]:.6f}")
print(f"الجولة ٢ | معامل النموذج: يدوياً {alpha2:.6f} | المكتبة {ada.estimator_weights_[1]:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

print("AdaBoostClassifier.get_params():", AdaBoostClassifier().get_params())

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=200, learning_rate=0.5, random_state=42,
).fit(X_tr, y_tr)

errs = ada.estimator_errors_
staged = list(ada.staged_score(X_te, y_te))
print(f"أخطاء الجذوع الأولى:  {np.round(errs[:5], 4)}")
print(f"أخطاء الجذوع الأخيرة: {np.round(errs[-3:], 4)}")
for k in [1, 10, 50, 100, 200]:
    print(f"بعد {k:3d} جولة | دقة الاختبار المتدرجة: {staged[k-1]:.4f}")
print(f"أفضل دقة متدرجة: {max(staged):.4f} عند الجولة رقم {int(np.argmax(staged)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1500], data.target[:1500], test_size=0.3, random_state=42
)

for loss in ["linear", "square", "exponential"]:
    reg = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=100, loss=loss, random_state=42,
    ).fit(X_tr, y_tr)
    rmse = root_mean_squared_error(y_te, reg.predict(X_te))
    print(f"loss={loss:12s} | RMSE الاختبار: {rmse:.4f} | R²: {reg.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

for depth, name in [(1, "جذع (عمق 1)"), (3, "شجرة عمق 3"), (None, "شجرة كاملة")]:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth, random_state=42),
        n_estimators=50, random_state=42,
    ).fit(X_tr, y_tr)
    print(f"قاعدة: {name:12s} | تدريب: {ada.score(X_tr, y_tr):.4f} | اختبار: {ada.score(X_te, y_te):.4f}")

single = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"شجرة مفردة كاملة بلا تعزيز | تدريب: {single.score(X_tr, y_tr):.4f} | اختبار: {single.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=600, n_features=10, n_informative=4,
    flip_y=0.20, class_sep=0.8, random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=150, random_state=42,
).fit(X_tr, y_tr)

errs = ada.estimator_errors_
print(f"أخطاء الجذوع الأولى: {np.round(errs[:4], 4)}")
print(f"أخطاء آخر 4 جولات:   {np.round(errs[-4:], 4)}  (حد العشوائية = 0.5)")
staged_tr = list(ada.staged_score(X_tr, y_tr))
staged_te = list(ada.staged_score(X_te, y_te))
for k in [5, 25, 50, 100, 150]:
    print(f"بعد {k:3d} جولة | تدريب: {staged_tr[k-1]:.4f} | اختبار: {staged_te[k-1]:.4f}")
print(f"أفضل دقة اختبار: {max(staged_te):.4f} عند الجولة {int(np.argmax(staged_te)) + 1}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=300, learning_rate=0.3, random_state=42,
).fit(X_tr, y_tr)

staged = list(ada.staged_score(X_te, y_te))
for k in [1, 10, 100]:
    print(f"الجولة {k:3d} | دقة الاختبار المتدرجة: {staged[k-1]:.4f}")
best_k = int(np.argmax(staged)) + 1
print(f"الجولة المثلى: {best_k} | دقتها: {staged[best_k-1]:.4f}")
print(f"النموذج الكامل (300 جولة): {ada.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-45-ch45.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
n, K = len(y), 3

w = np.full(n, 1.0 / n)
manual_eps, manual_alpha = [], []
for t in range(5):
    stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y, sample_weight=w)
    wrong = stump.predict(X) != y
    eps = np.average(wrong, weights=w)
    alpha = np.log((1 - eps) / eps) + np.log(K - 1)
    manual_eps.append(eps)
    manual_alpha.append(alpha)
    w = w * np.exp(alpha * wrong)
    w = w / w.sum()

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=5, random_state=42,
).fit(X, y)

print("الجولة | الخطأ يدوياً | الخطأ مكتبة | α يدوياً | α مكتبة")
for t in range(5):
    print(f"  {t+1}   |  {manual_eps[t]:.6f}  |  {ada.estimator_errors_[t]:.6f}  |"
          f" {manual_alpha[t]:.6f} | {ada.estimator_weights_[t]:.6f}")
diff_eps = np.max(np.abs(np.array(manual_eps) - ada.estimator_errors_))
diff_alpha = np.max(np.abs(np.array(manual_alpha) - ada.estimator_weights_))
print(f"أكبر فرق في الأخطاء: {diff_eps:.2e} | أكبر فرق في المعاملات: {diff_alpha:.2e}")
