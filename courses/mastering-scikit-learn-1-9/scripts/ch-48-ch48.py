#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصويت والتكديس
Mastering scikit-learn 1.9 — Chapter 48 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-48-ch48 (lesson 48)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-48-ch48.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصويت والتكديس
Mastering scikit-learn 1.9 — Chapter 48 — Ensembles

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-48-ch48 (lesson 48)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1500, n_features=15, n_informative=4,
                           n_redundant=2, n_clusters_per_class=3, flip_y=0.08,
                           weights=[0.85, 0.15], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

lr   = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
tree = DecisionTreeClassifier(max_depth=8, random_state=42)
knn  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))
estimators = [("lr", lr), ("tree", tree), ("knn", knn)]

for name, m in estimators:
    m.fit(X_tr, y_tr)
    print(f"النموذج المفرد {name:5s} | دقة الاختبار: {m.score(X_te, y_te):.4f}")

hard = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1).fit(X_tr, y_tr)
soft = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1).fit(X_tr, y_tr)
print(f"تصويت صلب hard | دقة الاختبار: {hard.score(X_te, y_te):.4f}")
print(f"تصويت ناعم soft | دقة الاختبار: {soft.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import VotingClassifier

for w in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]:
    vc = VotingClassifier(estimators=estimators, voting="soft",
                          weights=list(w), n_jobs=-1).fit(X_tr, y_tr)
    print(f"أوزان (lr, tree, knn) = {w} | دقة الاختبار: {vc.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto", random_state=42))
bad = VotingClassifier(estimators=[("svc", svc), ("tree", tree)], voting="soft")
bad.fit(X_tr, y_tr)  # التدريب يمر دون شكوى!
try:
    bad.predict(X_te)
except AttributeError as err:
    print(f"الانفجار يؤجل نفسه إلى predict: {type(err).__name__}")
    print(f"الرسالة: {err}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
Xr, yr = data.data[:3000], data.target[:3000]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
rknn  = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))
reg_estimators = [("ridge", ridge), ("tree", dtree), ("knn", rknn)]

for name, m in reg_estimators:
    m.fit(Xr_tr, yr_tr)
    print(f"النموذج المفرد {name:6s} | RMSE: {root_mean_squared_error(yr_te, m.predict(Xr_te)):.4f}")

vreg = VotingRegressor(estimators=reg_estimators, n_jobs=-1).fit(Xr_tr, yr_tr)
print(f"VotingRegressor (متوسط التنبؤات) | RMSE: {root_mean_squared_error(yr_te, vreg.predict(Xr_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

Xs, ys = make_classification(n_samples=1000, n_features=16, n_informative=6, random_state=42)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3,
                                              random_state=42, stratify=ys)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
stree = DecisionTreeClassifier(max_depth=8, random_state=42)
sknn  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))
stack_estimators = [("logit", logit), ("tree", stree), ("knn", sknn)]

for name, m in stack_estimators:
    m.fit(Xs_tr, ys_tr)
    print(f"النموذج المفرد {name:6s} | دقة الاختبار: {m.score(Xs_te, ys_te):.4f}")

stack = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"StackingClassifier (cv=5)   | دقة الاختبار: {stack.score(Xs_te, ys_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto", random_state=42))

# auto يلتقط decision_function تلقائياً من LinearSVC لأن predict_proba غير موجودة
stack_auto = StackingClassifier(
    estimators=[("svc", svc), ("tree", stree)],
    final_estimator=LogisticRegression(max_iter=2000),
    stack_method="auto", cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"تكديس (SVC + شجرة) بـ stack_method='auto' | دقة الاختبار: {stack_auto.score(Xs_te, ys_te):.4f}")

# passthrough: 16 ميزة أصلية + عمود تنبؤ واحد = 17 عموداً للمقدر النهائي
stack_pass = StackingClassifier(
    estimators=[("tree", stree)],
    final_estimator=LogisticRegression(max_iter=2000),
    passthrough=True, cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"أبعاد ميزات المقدر النهائي مع passthrough: {stack_pass.final_estimator_.coef_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.frozen import FrozenEstimator
import numpy as np

pred_before = stack.predict(Xs_te)
frozen_stack = FrozenEstimator(stack)          # المجموعة كاملة مجمدة الآن
frozen_stack.fit(Xs_tr, ys_tr)                 # لا تقلق: هذا لا شيء فعلياً
pred_after = frozen_stack.predict(Xs_te)
print(f"التطابق قبل وبعد fit الوهمي: {np.array_equal(pred_before, pred_after)}")
print(f"الدقة تبقى: {frozen_stack.score(Xs_te, ys_te):.4f}")
print(f"الكائن المجمد يلف المجموعة نفسها دون نسخ: {frozen_stack.estimator is stack}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import StackingRegressor

sreg = StackingRegressor(
    estimators=reg_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5, n_jobs=-1
).fit(Xr_tr, yr_tr)
print(f"VotingRegressor  | RMSE: {root_mean_squared_error(yr_te, vreg.predict(Xr_te)):.4f}")
print(f"StackingRegressor| RMSE: {root_mean_squared_error(yr_te, sreg.predict(Xr_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

rf1 = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1)
rf2 = RandomForestClassifier(n_estimators=50, random_state=2, n_jobs=-1)

stack_trees = StackingClassifier(
    estimators=[("rf1", rf1), ("rf2", rf2)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)

stack_mixed = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)

print(f"تكديس غابتين متشابهتين | دقة الاختبار: {stack_trees.score(Xs_te, ys_te):.4f}")
print(f"تكديس ثلاثي متنوع      | دقة الاختبار: {stack_mixed.score(Xs_te, ys_te):.4f}")
print(f"أفضل فرد (knn)         | دقة الاختبار: {sknn.score(Xs_te, ys_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Xw, yw = load_wine(return_X_y=True)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.3,
                                              random_state=42, stratify=yw)

wine_estimators = [
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("tree", make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=5, random_state=42))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9))),
]
for name, m in wine_estimators:
    m.fit(Xw_tr, yw_tr)
    print(f"الفرد {name:5s} | الدقة: {m.score(Xw_te, yw_te):.4f}")

for voting in ["hard", "soft"]:
    vc = VotingClassifier(estimators=wine_estimators, voting=voting, n_jobs=-1).fit(Xw_tr, yw_tr)
    print(f"تصويت {voting:4s} | الدقة: {vc.score(Xw_te, yw_te):.4f}")

best = (None, -1.0)
for w in [(1, 1, 1), (2, 1, 1), (1, 1, 2)]:
    vc = VotingClassifier(estimators=wine_estimators, voting="soft",
                          weights=list(w), n_jobs=-1).fit(Xw_tr, yw_tr)
    acc = vc.score(Xw_te, yw_te)
    print(f"soft بأوزان {w} | الدقة: {acc:.4f}")
    if acc > best[1]:
        best = (w, acc)
print(f"أفضل شبكة أوزان: {best[0]} بدقة {best[1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [011] content/part06/ch48.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
Xr, yr = data.data[:3000], data.target[:3000]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

base = [
    ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("tree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))),
]
for pass_flag in [False, True]:
    sr = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0),
                           passthrough=pass_flag, cv=5, n_jobs=-1).fit(Xr_tr, yr_tr)
    rmse = root_mean_squared_error(yr_te, sr.predict(Xr_te))
    print(f"passthrough={pass_flag} | RMSE: {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1500, n_features=15, n_informative=4,
                           n_redundant=2, n_clusters_per_class=3, flip_y=0.08,
                           weights=[0.85, 0.15], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

lr   = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
tree = DecisionTreeClassifier(max_depth=8, random_state=42)
knn  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))
estimators = [("lr", lr), ("tree", tree), ("knn", knn)]

for name, m in estimators:
    m.fit(X_tr, y_tr)
    print(f"النموذج المفرد {name:5s} | دقة الاختبار: {m.score(X_te, y_te):.4f}")

hard = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1).fit(X_tr, y_tr)
soft = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1).fit(X_tr, y_tr)
print(f"تصويت صلب hard | دقة الاختبار: {hard.score(X_te, y_te):.4f}")
print(f"تصويت ناعم soft | دقة الاختبار: {soft.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import VotingClassifier

for w in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]:
    vc = VotingClassifier(estimators=estimators, voting="soft",
                          weights=list(w), n_jobs=-1).fit(X_tr, y_tr)
    print(f"أوزان (lr, tree, knn) = {w} | دقة الاختبار: {vc.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto", random_state=42))
bad = VotingClassifier(estimators=[("svc", svc), ("tree", tree)], voting="soft")
bad.fit(X_tr, y_tr)  # التدريب يمر دون شكوى!
try:
    bad.predict(X_te)
except AttributeError as err:
    print(f"الانفجار يؤجل نفسه إلى predict: {type(err).__name__}")
    print(f"الرسالة: {err}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
Xr, yr = data.data[:3000], data.target[:3000]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
rknn  = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))
reg_estimators = [("ridge", ridge), ("tree", dtree), ("knn", rknn)]

for name, m in reg_estimators:
    m.fit(Xr_tr, yr_tr)
    print(f"النموذج المفرد {name:6s} | RMSE: {root_mean_squared_error(yr_te, m.predict(Xr_te)):.4f}")

vreg = VotingRegressor(estimators=reg_estimators, n_jobs=-1).fit(Xr_tr, yr_tr)
print(f"VotingRegressor (متوسط التنبؤات) | RMSE: {root_mean_squared_error(yr_te, vreg.predict(Xr_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

Xs, ys = make_classification(n_samples=1000, n_features=16, n_informative=6, random_state=42)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3,
                                              random_state=42, stratify=ys)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
stree = DecisionTreeClassifier(max_depth=8, random_state=42)
sknn  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))
stack_estimators = [("logit", logit), ("tree", stree), ("knn", sknn)]

for name, m in stack_estimators:
    m.fit(Xs_tr, ys_tr)
    print(f"النموذج المفرد {name:6s} | دقة الاختبار: {m.score(Xs_te, ys_te):.4f}")

stack = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"StackingClassifier (cv=5)   | دقة الاختبار: {stack.score(Xs_te, ys_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto", random_state=42))

# auto يلتقط decision_function تلقائياً من LinearSVC لأن predict_proba غير موجودة
stack_auto = StackingClassifier(
    estimators=[("svc", svc), ("tree", stree)],
    final_estimator=LogisticRegression(max_iter=2000),
    stack_method="auto", cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"تكديس (SVC + شجرة) بـ stack_method='auto' | دقة الاختبار: {stack_auto.score(Xs_te, ys_te):.4f}")

# passthrough: 16 ميزة أصلية + عمود تنبؤ واحد = 17 عموداً للمقدر النهائي
stack_pass = StackingClassifier(
    estimators=[("tree", stree)],
    final_estimator=LogisticRegression(max_iter=2000),
    passthrough=True, cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)
print(f"أبعاد ميزات المقدر النهائي مع passthrough: {stack_pass.final_estimator_.coef_.shape}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.frozen import FrozenEstimator
import numpy as np

pred_before = stack.predict(Xs_te)
frozen_stack = FrozenEstimator(stack)          # المجموعة كاملة مجمدة الآن
frozen_stack.fit(Xs_tr, ys_tr)                 # لا تقلق: هذا لا شيء فعلياً
pred_after = frozen_stack.predict(Xs_te)
print(f"التطابق قبل وبعد fit الوهمي: {np.array_equal(pred_before, pred_after)}")
print(f"الدقة تبقى: {frozen_stack.score(Xs_te, ys_te):.4f}")
print(f"الكائن المجمد يلف المجموعة نفسها دون نسخ: {frozen_stack.estimator is stack}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import StackingRegressor

sreg = StackingRegressor(
    estimators=reg_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5, n_jobs=-1
).fit(Xr_tr, yr_tr)
print(f"VotingRegressor  | RMSE: {root_mean_squared_error(yr_te, vreg.predict(Xr_te)):.4f}")
print(f"StackingRegressor| RMSE: {root_mean_squared_error(yr_te, sreg.predict(Xr_te)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

rf1 = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1)
rf2 = RandomForestClassifier(n_estimators=50, random_state=2, n_jobs=-1)

stack_trees = StackingClassifier(
    estimators=[("rf1", rf1), ("rf2", rf2)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)

stack_mixed = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1
).fit(Xs_tr, ys_tr)

print(f"تكديس غابتين متشابهتين | دقة الاختبار: {stack_trees.score(Xs_te, ys_te):.4f}")
print(f"تكديس ثلاثي متنوع      | دقة الاختبار: {stack_mixed.score(Xs_te, ys_te):.4f}")
print(f"أفضل فرد (knn)         | دقة الاختبار: {sknn.score(Xs_te, ys_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Xw, yw = load_wine(return_X_y=True)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.3,
                                              random_state=42, stratify=yw)

wine_estimators = [
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("tree", make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=5, random_state=42))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9))),
]
for name, m in wine_estimators:
    m.fit(Xw_tr, yw_tr)
    print(f"الفرد {name:5s} | الدقة: {m.score(Xw_te, yw_te):.4f}")

for voting in ["hard", "soft"]:
    vc = VotingClassifier(estimators=wine_estimators, voting=voting, n_jobs=-1).fit(Xw_tr, yw_tr)
    print(f"تصويت {voting:4s} | الدقة: {vc.score(Xw_te, yw_te):.4f}")

best = (None, -1.0)
for w in [(1, 1, 1), (2, 1, 1), (1, 1, 2)]:
    vc = VotingClassifier(estimators=wine_estimators, voting="soft",
                          weights=list(w), n_jobs=-1).fit(Xw_tr, yw_tr)
    acc = vc.score(Xw_te, yw_te)
    print(f"soft بأوزان {w} | الدقة: {acc:.4f}")
    if acc > best[1]:
        best = (w, acc)
print(f"أفضل شبكة أوزان: {best[0]} بدقة {best[1]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-48-ch48.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = fetch_california_housing()
Xr, yr = data.data[:3000], data.target[:3000]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

base = [
    ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("tree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))),
]
for pass_flag in [False, True]:
    sr = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0),
                           passthrough=pass_flag, cv=5, n_jobs=-1).fit(Xr_tr, yr_tr)
    rmse = root_mean_squared_error(yr_te, sr.predict(Xr_te))
    print(f"passthrough={pass_flag} | RMSE: {rmse:.4f}")
