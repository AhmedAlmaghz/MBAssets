#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قياس أهمية الخصائص: ثلاث طرق مختلفة
Mastering scikit-learn 1.9 — Chapter 25 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-25-ch25 (lesson 25)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-25-ch25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — قياس أهمية الخصائص: ثلاث طرق مختلفة
Mastering scikit-learn 1.9 — Chapter 25 — Features

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-25-ch25 (lesson 25)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("أهمية الميزات (Impurity-Based) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_names[idx]:<35} {importances[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(15), importances[indices])
ax.set_xticks(range(15))
ax.set_xticklabels([feature_names[i][:15] for i in indices], rotation=45, ha="right")
ax.set_xlabel("Feature")
ax.set_ylabel("Impurity Importance")
ax.set_title("Random Forest — Feature Importances (Impurity-Based)")
plt.tight_layout()
plt.savefig("importance_impurity.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [003] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe.fit(X, y)

coefs = pipe.named_steps["clf"].coef_[0]
indices = np.argsort(np.abs(coefs))[::-1]

print("أهمية المعاملات الخطية (|coef_|) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    direction = "↑" if coefs[idx] > 0 else "↓"
    print(f"  {i+1:2d}. {direction} {feature_names[idx]:<35} {coefs[idx]:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

perm_imp = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

indices = np.argsort(perm_imp.importances_mean)[::-1]
print("أهمية الإبدال (Permutation Importance) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_names[idx]:<35} "
          f"{perm_imp.importances_mean[idx]:.4f} ± {perm_imp.importances_std[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1. Impurity (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
imp_impurity = rf.feature_importances_

# 2. Permutation (on RF)
perm = permutation_importance(rf, X_test, y_test,
                               n_repeats=10, random_state=42, n_jobs=-1)
imp_perm = perm.importances_mean

# 3. Linear coef
pipe_lr = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe_lr.fit(X_train, y_train)
imp_coef = np.abs(pipe_lr.named_steps["lr"].coef_[0])
# normalise for comparison
imp_coef = imp_coef / imp_coef.sum()
imp_perm_norm = np.clip(imp_perm, 0, None)
imp_perm_norm = imp_perm_norm / (imp_perm_norm.sum() + 1e-12)

# أعلى 5 وفق كل طريقة
top5_impurity = set(np.argsort(imp_impurity)[::-1][:5])
top5_perm = set(np.argsort(imp_perm)[::-1][:5])
top5_coef = set(np.argsort(imp_coef)[::-1][:5])

common_all = top5_impurity & top5_perm & top5_coef
print("أعلى 5 وفق Impurity:", [feature_names[i] for i in top5_impurity])
print("أعلى 5 وفق Permutation:", [feature_names[i] for i in top5_perm])
print("أعلى 5 وفق |coef|:", [feature_names[i] for i in top5_coef])
print("مشتركة بين الثلاث:", [feature_names[i] for i in common_all])

# ────────────────────────────────────────────────────────────────────────
# [006] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
cum_importance = np.cumsum(importances[indices])

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(range(len(importances)), importances[indices], alpha=0.7)
ax1.set_xlabel("Feature rank")
ax1.set_ylabel("Impurity Importance")
ax2 = ax1.twinx()
ax2.plot(range(len(importances)), cum_importance * 100, color="red",
         linewidth=2, label="Cumulative %")
ax2.axhline(80, color="gray", linestyle="--", alpha=0.6)
ax2.set_ylabel("Cumulative Importance (%)")
ax2.legend(loc="lower right")
ax1.set_title("Pareto Chart — Feature Importances")
plt.tight_layout()
plt.savefig("pareto_importance.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [007] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("أهمية الميزات (California Housing):")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]:<15} {importances[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part03/ch25.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)

perm = permutation_importance(rf, X_te, y_te,
                               n_repeats=10, random_state=42, n_jobs=-1)
idx_perm = np.argsort(perm.importances_mean)[::-1]
idx_imp = np.argsort(rf.feature_importances_)[::-1]

print("ترتيب Impurity:    ", [feature_names[i] for i in idx_imp])
print("ترتيب Permutation: ", [feature_names[i] for i in idx_perm])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("أهمية الميزات (Impurity-Based) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_names[idx]:<35} {importances[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(15), importances[indices])
ax.set_xticks(range(15))
ax.set_xticklabels([feature_names[i][:15] for i in indices], rotation=45, ha="right")
ax.set_xlabel("Feature")
ax.set_ylabel("Impurity Importance")
ax.set_title("Random Forest — Feature Importances (Impurity-Based)")
plt.tight_layout()
plt.savefig("importance_impurity.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe.fit(X, y)

coefs = pipe.named_steps["clf"].coef_[0]
indices = np.argsort(np.abs(coefs))[::-1]

print("أهمية المعاملات الخطية (|coef_|) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    direction = "↑" if coefs[idx] > 0 else "↓"
    print(f"  {i+1:2d}. {direction} {feature_names[idx]:<35} {coefs[idx]:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

perm_imp = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

indices = np.argsort(perm_imp.importances_mean)[::-1]
print("أهمية الإبدال (Permutation Importance) — أعلى 10:")
for i in range(10):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_names[idx]:<35} "
          f"{perm_imp.importances_mean[idx]:.4f} ± {perm_imp.importances_std[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1. Impurity (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
imp_impurity = rf.feature_importances_

# 2. Permutation (on RF)
perm = permutation_importance(rf, X_test, y_test,
                               n_repeats=10, random_state=42, n_jobs=-1)
imp_perm = perm.importances_mean

# 3. Linear coef
pipe_lr = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, max_iter=5000, random_state=42))
])
pipe_lr.fit(X_train, y_train)
imp_coef = np.abs(pipe_lr.named_steps["lr"].coef_[0])
# normalise for comparison
imp_coef = imp_coef / imp_coef.sum()
imp_perm_norm = np.clip(imp_perm, 0, None)
imp_perm_norm = imp_perm_norm / (imp_perm_norm.sum() + 1e-12)

# أعلى 5 وفق كل طريقة
top5_impurity = set(np.argsort(imp_impurity)[::-1][:5])
top5_perm = set(np.argsort(imp_perm)[::-1][:5])
top5_coef = set(np.argsort(imp_coef)[::-1][:5])

common_all = top5_impurity & top5_perm & top5_coef
print("أعلى 5 وفق Impurity:", [feature_names[i] for i in top5_impurity])
print("أعلى 5 وفق Permutation:", [feature_names[i] for i in top5_perm])
print("أعلى 5 وفق |coef|:", [feature_names[i] for i in top5_coef])
print("مشتركة بين الثلاث:", [feature_names[i] for i in common_all])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
cum_importance = np.cumsum(importances[indices])

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(range(len(importances)), importances[indices], alpha=0.7)
ax1.set_xlabel("Feature rank")
ax1.set_ylabel("Impurity Importance")
ax2 = ax1.twinx()
ax2.plot(range(len(importances)), cum_importance * 100, color="red",
         linewidth=2, label="Cumulative %")
ax2.axhline(80, color="gray", linestyle="--", alpha=0.6)
ax2.set_ylabel("Cumulative Importance (%)")
ax2.legend(loc="lower right")
ax1.set_title("Pareto Chart — Feature Importances")
plt.tight_layout()
plt.savefig("pareto_importance.png", dpi=100)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("أهمية الميزات (California Housing):")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]:<15} {importances[idx]:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-25-ch25.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)

perm = permutation_importance(rf, X_te, y_te,
                               n_repeats=10, random_state=42, n_jobs=-1)
idx_perm = np.argsort(perm.importances_mean)[::-1]
idx_imp = np.argsort(rf.feature_importances_)[::-1]

print("ترتيب Impurity:    ", [feature_names[i] for i in idx_imp])
print("ترتيب Permutation: ", [feature_names[i] for i in idx_perm])
