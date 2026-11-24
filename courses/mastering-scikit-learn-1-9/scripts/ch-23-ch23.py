#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصفية التي تنظر إلى النموذج
Mastering scikit-learn 1.9 — Chapter 23 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-23-ch23 (lesson 23)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-23-ch23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التصفية التي تنظر إلى النموذج
Mastering scikit-learn 1.9 — Chapter 23 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-23-ch23 (lesson 23)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
print(f"الأبعاد الأصلية: {X.shape}")

# ندرّب غابة عشوائية ونستخدمها لاختيار الميزات
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold="mean")
selector.fit(X, y)

X_sel = selector.transform(X)
print(f"الأبعاد بعد SelectFromModel (threshold=mean): {X_sel.shape}")
print(f"مؤشرات الميزات المختارة: {selector.get_support(indices=True)}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold="median")
selector.fit(X, y)

chosen = feature_names[selector.get_support()]
print("الميزات المختارة بعتبة المتوسط:")
for name in chosen:
    print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

# نستخدم الانحدار اللوجيستي (C صغير لزيادة التنظيم)
lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
rfe = RFE(estimator=lr, n_features_to_select=10, step=1)
rfe.fit(X, y)

print(f"الأبعاد الأصلية: {X.shape[1]}")
print(f"الأبعاد المختارة: {rfe.n_features_}")
print("\nترتيب الميزات (1 = مختارة):")
for name, rank, support in zip(feature_names, rfe.ranking_, rfe.support_):
    status = "✓" if support else f"رتبة {rank}"
    print(f"  {status}  {name}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=lr,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X, y)

print(f"العدد الأمثل من الميزات: {rfecv.n_features_}")
print(f"\nالميزات المختارة:")
for name, support in zip(feature_names, rfecv.support_):
    if support:
        print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(estimator=lr, step=1, cv=cv, scoring="accuracy",
              min_features_to_select=5, n_jobs=-1)
rfecv.fit(X, y)

n_scores = len(rfecv.cv_results_["mean_test_score"])
plt.figure(figsize=(9, 4))
plt.plot(range(rfecv.min_features_to_select, n_scores + rfecv.min_features_to_select),
         rfecv.cv_results_["mean_test_score"], marker="o", markersize=4)
plt.axvline(rfecv.n_features_, color="red", linestyle="--",
            label=f"الأمثل = {rfecv.n_features_}")
plt.xlabel("Number of features selected")
plt.ylabel("Mean CV accuracy")
plt.title("RFECV — Cross-validation Accuracy vs Number of Features")
plt.legend()
plt.tight_layout()
plt.savefig("rfecv_curve.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [006] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

knn = KNeighborsClassifier(n_neighbors=5)

# الاتجاه الأمامي: اختر أفضل ميزتين
sfs = SequentialFeatureSelector(
    knn,
    n_features_to_select=2,
    direction="forward",
    scoring="accuracy",
    cv=5
)
sfs.fit(X, y)

chosen = [feature_names[i] for i in sfs.get_support(indices=True)]
print("أفضل ميزتين بالاتجاه الأمامي:")
for f in chosen:
    print(f"  ✓ {f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel, RFE, RFECV, SequentialFeatureSelector
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

results = {}

# 1. SelectFromModel
sfm = SelectFromModel(rf, threshold="mean")
sfm.fit(X, y)
X_sfm = sfm.transform(X)
score_sfm = cross_val_score(lr, X_sfm, y, cv=cv, scoring="accuracy").mean()
results["SelectFromModel"] = (X_sfm.shape[1], score_sfm)

# 2. RFE (10 ميزات)
rfe = RFE(lr, n_features_to_select=10, step=1)
rfe.fit(X, y)
X_rfe = rfe.transform(X)
score_rfe = cross_val_score(lr, X_rfe, y, cv=cv, scoring="accuracy").mean()
results["RFE (10)"] = (X_rfe.shape[1], score_rfe)

# 3. RFECV
rfecv = RFECV(lr, step=1, cv=cv, scoring="accuracy",
              min_features_to_select=5, n_jobs=-1)
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
score_rfecv = cross_val_score(lr, X_rfecv, y, cv=cv, scoring="accuracy").mean()
results["RFECV"] = (X_rfecv.shape[1], score_rfecv)

# 4. SequentialFeatureSelector
sfs = SequentialFeatureSelector(lr, n_features_to_select=10,
                                direction="forward", cv=cv)
sfs.fit(X, y)
X_sfs = sfs.transform(X)
score_sfs = cross_val_score(lr, X_sfs, y, cv=cv, scoring="accuracy").mean()
results["SFS (forward, 10)"] = (X_sfs.shape[1], score_sfs)

# 5. Baseline (كل الميزات)
score_all = cross_val_score(lr, X, y, cv=cv, scoring="accuracy").mean()

print(f"{'الأسلوب':<25} {'عدد الميزات':>12} {'دقة CV':>10}")
print("-" * 50)
print(f"{'الكل (30 ميزة)':<25} {'30':>12} {score_all:.4f}")
for name, (n_feat, score) in results.items():
    print(f"{name:<25} {n_feat:>12} {score:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_wine(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)

# الخط الأساسي
score_base = cross_val_score(lr, X, y, cv=cv, scoring="accuracy").mean()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold="median")
sfm.fit(X, y)
X_sel = sfm.transform(X)
score_sel = cross_val_score(lr, X_sel, y, cv=cv, scoring="accuracy").mean()

print(f"الميزات الأصلية: 13  → دقة CV: {score_base:.4f}")
print(f"الميزات بعد SelectFromModel(median): {X_sel.shape[1]}  → دقة CV: {score_sel:.4f}")
delta = score_sel - score_base
print(f"التغيّر: {delta:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part03/ch23.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
feature_names = list(load_iris().feature_names)
knn = KNeighborsClassifier(n_neighbors=5)

for direction in ["forward", "backward"]:
    sfs = SequentialFeatureSelector(knn, n_features_to_select=2,
                                    direction=direction, cv=5)
    sfs.fit(X, y)
    chosen = [feature_names[i] for i in sfs.get_support(indices=True)]
    X_sel = sfs.transform(X)
    score = cross_val_score(knn, X_sel, y, cv=5, scoring="accuracy").mean()
    print(f"{direction:>8}: {chosen}  → دقة: {score:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
print(f"الأبعاد الأصلية: {X.shape}")

# ندرّب غابة عشوائية ونستخدمها لاختيار الميزات
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold="mean")
selector.fit(X, y)

X_sel = selector.transform(X)
print(f"الأبعاد بعد SelectFromModel (threshold=mean): {X_sel.shape}")
print(f"مؤشرات الميزات المختارة: {selector.get_support(indices=True)}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold="median")
selector.fit(X, y)

chosen = feature_names[selector.get_support()]
print("الميزات المختارة بعتبة المتوسط:")
for name in chosen:
    print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

# نستخدم الانحدار اللوجيستي (C صغير لزيادة التنظيم)
lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
rfe = RFE(estimator=lr, n_features_to_select=10, step=1)
rfe.fit(X, y)

print(f"الأبعاد الأصلية: {X.shape[1]}")
print(f"الأبعاد المختارة: {rfe.n_features_}")
print("\nترتيب الميزات (1 = مختارة):")
for name, rank, support in zip(feature_names, rfe.ranking_, rfe.support_):
    status = "✓" if support else f"رتبة {rank}"
    print(f"  {status}  {name}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=lr,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X, y)

print(f"العدد الأمثل من الميزات: {rfecv.n_features_}")
print(f"\nالميزات المختارة:")
for name, support in zip(feature_names, rfecv.support_):
    if support:
        print(f"  ✓ {name}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(estimator=lr, step=1, cv=cv, scoring="accuracy",
              min_features_to_select=5, n_jobs=-1)
rfecv.fit(X, y)

n_scores = len(rfecv.cv_results_["mean_test_score"])
plt.figure(figsize=(9, 4))
plt.plot(range(rfecv.min_features_to_select, n_scores + rfecv.min_features_to_select),
         rfecv.cv_results_["mean_test_score"], marker="o", markersize=4)
plt.axvline(rfecv.n_features_, color="red", linestyle="--",
            label=f"الأمثل = {rfecv.n_features_}")
plt.xlabel("Number of features selected")
plt.ylabel("Mean CV accuracy")
plt.title("RFECV — Cross-validation Accuracy vs Number of Features")
plt.legend()
plt.tight_layout()
plt.savefig("rfecv_curve.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

knn = KNeighborsClassifier(n_neighbors=5)

# الاتجاه الأمامي: اختر أفضل ميزتين
sfs = SequentialFeatureSelector(
    knn,
    n_features_to_select=2,
    direction="forward",
    scoring="accuracy",
    cv=5
)
sfs.fit(X, y)

chosen = [feature_names[i] for i in sfs.get_support(indices=True)]
print("أفضل ميزتين بالاتجاه الأمامي:")
for f in chosen:
    print(f"  ✓ {f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel, RFE, RFECV, SequentialFeatureSelector
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(C=0.1, max_iter=5000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

results = {}

# 1. SelectFromModel
sfm = SelectFromModel(rf, threshold="mean")
sfm.fit(X, y)
X_sfm = sfm.transform(X)
score_sfm = cross_val_score(lr, X_sfm, y, cv=cv, scoring="accuracy").mean()
results["SelectFromModel"] = (X_sfm.shape[1], score_sfm)

# 2. RFE (10 ميزات)
rfe = RFE(lr, n_features_to_select=10, step=1)
rfe.fit(X, y)
X_rfe = rfe.transform(X)
score_rfe = cross_val_score(lr, X_rfe, y, cv=cv, scoring="accuracy").mean()
results["RFE (10)"] = (X_rfe.shape[1], score_rfe)

# 3. RFECV
rfecv = RFECV(lr, step=1, cv=cv, scoring="accuracy",
              min_features_to_select=5, n_jobs=-1)
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
score_rfecv = cross_val_score(lr, X_rfecv, y, cv=cv, scoring="accuracy").mean()
results["RFECV"] = (X_rfecv.shape[1], score_rfecv)

# 4. SequentialFeatureSelector
sfs = SequentialFeatureSelector(lr, n_features_to_select=10,
                                direction="forward", cv=cv)
sfs.fit(X, y)
X_sfs = sfs.transform(X)
score_sfs = cross_val_score(lr, X_sfs, y, cv=cv, scoring="accuracy").mean()
results["SFS (forward, 10)"] = (X_sfs.shape[1], score_sfs)

# 5. Baseline (كل الميزات)
score_all = cross_val_score(lr, X, y, cv=cv, scoring="accuracy").mean()

print(f"{'الأسلوب':<25} {'عدد الميزات':>12} {'دقة CV':>10}")
print("-" * 50)
print(f"{'الكل (30 ميزة)':<25} {'30':>12} {score_all:.4f}")
for name, (n_feat, score) in results.items():
    print(f"{name:<25} {n_feat:>12} {score:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_wine(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)

# الخط الأساسي
score_base = cross_val_score(lr, X, y, cv=cv, scoring="accuracy").mean()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold="median")
sfm.fit(X, y)
X_sel = sfm.transform(X)
score_sel = cross_val_score(lr, X_sel, y, cv=cv, scoring="accuracy").mean()

print(f"الميزات الأصلية: 13  → دقة CV: {score_base:.4f}")
print(f"الميزات بعد SelectFromModel(median): {X_sel.shape[1]}  → دقة CV: {score_sel:.4f}")
delta = score_sel - score_base
print(f"التغيّر: {delta:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-23-ch23.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
feature_names = list(load_iris().feature_names)
knn = KNeighborsClassifier(n_neighbors=5)

for direction in ["forward", "backward"]:
    sfs = SequentialFeatureSelector(knn, n_features_to_select=2,
                                    direction=direction, cv=5)
    sfs.fit(X, y)
    chosen = [feature_names[i] for i in sfs.get_support(indices=True)]
    X_sel = sfs.transform(X)
    score = cross_val_score(knn, X_sel, y, cv=5, scoring="accuracy").mean()
    print(f"{direction:>8}: {chosen}  → دقة: {score:.4f}")
