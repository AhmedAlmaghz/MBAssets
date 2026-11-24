#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كشف الشذوذ وتقدير التباين
Mastering scikit-learn 1.9 — Chapter 70 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-70-ch70 (lesson 70)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-70-ch70.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كشف الشذوذ وتقدير التباين
Mastering scikit-learn 1.9 — Chapter 70 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-70-ch70 (lesson 70)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
X_train = rng.normal(size=(500, 2))
X_test = np.vstack([rng.normal(size=(40, 2)),
                    [[4.5, 4.5], [-4.0, 4.0], [0.2, -5.0]]])

model = IsolationForest(n_estimators=200, contamination=0.06,
                        random_state=42).fit(X_train)
labels = model.predict(X_test)
scores = model.decision_function(X_test)

print("عدد الحالات المصنفة شاذة:", int((labels == -1).sum()))
print("الطبيعي يحمل 1 والشاذ يحمل -1")
print("أقل خمس درجات؛ الأصغر أشد شذوذاً:",
      np.round(np.sort(scores)[:5], 3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(7)
dense = rng.normal(loc=[0, 0], scale=0.35, size=(180, 2))
small_cluster = rng.normal(loc=[4, 4], scale=0.22, size=(35, 2))
X = np.vstack([dense, small_cluster, [[0.9, 0.9]]])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
labels = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

idx = np.argsort(scores)[:5]
print("عدد الشواذ:", int((labels == -1).sum()))
print("أشد الحالات شذوذاً (الفهرس، الدرجة):")
print([(int(i), round(float(scores[i]), 2)) for i in idx])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(12)
X_train = rng.normal(size=(700, 2))
X_normal = rng.normal(size=(180, 2))
X_outlier = rng.normal(loc=[3.8, 3.8], scale=0.55, size=(20, 2))
X_test = np.vstack([X_normal, X_outlier])
y_test = np.r_[np.zeros(len(X_normal)), np.ones(len(X_outlier))]

models = {
    "IsolationForest": IsolationForest(contamination=0.05, random_state=12),
    "LOF novelty": LocalOutlierFactor(n_neighbors=25, novelty=True,
                                      contamination=0.05),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.05, random_state=12),
}
for name, model in models.items():
    model.fit(X_train)
    anomaly_score = -model.decision_function(X_test)
    print(f"{name:18s} ROC AUC={roc_auc_score(y_test, anomaly_score):.3f} | "
          f"AP={average_precision_score(y_test, anomaly_score):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(17)
X_train = rng.normal(size=(500, 2)) * [1.0, 100.0]
X_test = np.vstack([
    rng.normal(size=(40, 2)) * [1.0, 100.0],
    [[4.5, 450.0], [-4.0, -420.0]],
])

model = make_pipeline(
    StandardScaler(),
    OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
).fit(X_train)
labels = model.predict(X_test)
scores = model.decision_function(X_test)
print("الشواذ المكتشفة:", int((labels == -1).sum()))
print("أدنى درجات القرار:", np.round(np.sort(scores)[:5], 3))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS

rng = np.random.default_rng(24)
latent = rng.normal(size=(45, 4))
mixing = rng.normal(size=(4, 18))
X = latent @ mixing + 0.03 * rng.normal(size=(45, 18))

estimators = {
    "Empirical": EmpiricalCovariance(),
    "LedoitWolf": LedoitWolf(),
    "OAS": OAS(),
}
for name, estimator in estimators.items():
    estimator.fit(X)
    condition = np.linalg.cond(estimator.covariance_)
    shrinkage = getattr(estimator, "shrinkage_", float("nan"))
    print(f"{name:12s} condition={condition:9.1f} | shrinkage={shrinkage:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(5)
X = np.vstack([rng.normal(size=(190, 2)),
               rng.uniform(-5, 5, size=(10, 2))])
model = IsolationForest(contamination=0.05, random_state=5).fit(X)
labels = model.predict(X)
print("الحالات المصنفة شاذة:", int((labels == -1).sum()))
print("تقدير الصحة يتطلب وسوماً مستقلة، لا قيمة contamination وحدها.")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch70.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

rng = np.random.default_rng(8)
X = rng.normal(size=(16, 30))
for estimator in (EmpiricalCovariance(), LedoitWolf()):
    estimator.fit(X)
    print(type(estimator).__name__,
          "rank=", np.linalg.matrix_rank(estimator.covariance_),
          "condition=", round(np.linalg.cond(estimator.covariance_), 2))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
X_train = rng.normal(size=(500, 2))
X_test = np.vstack([rng.normal(size=(40, 2)),
                    [[4.5, 4.5], [-4.0, 4.0], [0.2, -5.0]]])

model = IsolationForest(n_estimators=200, contamination=0.06,
                        random_state=42).fit(X_train)
labels = model.predict(X_test)
scores = model.decision_function(X_test)

print("عدد الحالات المصنفة شاذة:", int((labels == -1).sum()))
print("الطبيعي يحمل 1 والشاذ يحمل -1")
print("أقل خمس درجات؛ الأصغر أشد شذوذاً:",
      np.round(np.sort(scores)[:5], 3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(7)
dense = rng.normal(loc=[0, 0], scale=0.35, size=(180, 2))
small_cluster = rng.normal(loc=[4, 4], scale=0.22, size=(35, 2))
X = np.vstack([dense, small_cluster, [[0.9, 0.9]]])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
labels = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

idx = np.argsort(scores)[:5]
print("عدد الشواذ:", int((labels == -1).sum()))
print("أشد الحالات شذوذاً (الفهرس، الدرجة):")
print([(int(i), round(float(scores[i]), 2)) for i in idx])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(12)
X_train = rng.normal(size=(700, 2))
X_normal = rng.normal(size=(180, 2))
X_outlier = rng.normal(loc=[3.8, 3.8], scale=0.55, size=(20, 2))
X_test = np.vstack([X_normal, X_outlier])
y_test = np.r_[np.zeros(len(X_normal)), np.ones(len(X_outlier))]

models = {
    "IsolationForest": IsolationForest(contamination=0.05, random_state=12),
    "LOF novelty": LocalOutlierFactor(n_neighbors=25, novelty=True,
                                      contamination=0.05),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.05, random_state=12),
}
for name, model in models.items():
    model.fit(X_train)
    anomaly_score = -model.decision_function(X_test)
    print(f"{name:18s} ROC AUC={roc_auc_score(y_test, anomaly_score):.3f} | "
          f"AP={average_precision_score(y_test, anomaly_score):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(17)
X_train = rng.normal(size=(500, 2)) * [1.0, 100.0]
X_test = np.vstack([
    rng.normal(size=(40, 2)) * [1.0, 100.0],
    [[4.5, 450.0], [-4.0, -420.0]],
])

model = make_pipeline(
    StandardScaler(),
    OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
).fit(X_train)
labels = model.predict(X_test)
scores = model.decision_function(X_test)
print("الشواذ المكتشفة:", int((labels == -1).sum()))
print("أدنى درجات القرار:", np.round(np.sort(scores)[:5], 3))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS

rng = np.random.default_rng(24)
latent = rng.normal(size=(45, 4))
mixing = rng.normal(size=(4, 18))
X = latent @ mixing + 0.03 * rng.normal(size=(45, 18))

estimators = {
    "Empirical": EmpiricalCovariance(),
    "LedoitWolf": LedoitWolf(),
    "OAS": OAS(),
}
for name, estimator in estimators.items():
    estimator.fit(X)
    condition = np.linalg.cond(estimator.covariance_)
    shrinkage = getattr(estimator, "shrinkage_", float("nan"))
    print(f"{name:12s} condition={condition:9.1f} | shrinkage={shrinkage:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(5)
X = np.vstack([rng.normal(size=(190, 2)),
               rng.uniform(-5, 5, size=(10, 2))])
model = IsolationForest(contamination=0.05, random_state=5).fit(X)
labels = model.predict(X)
print("الحالات المصنفة شاذة:", int((labels == -1).sum()))
print("تقدير الصحة يتطلب وسوماً مستقلة، لا قيمة contamination وحدها.")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-70-ch70.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

rng = np.random.default_rng(8)
X = rng.normal(size=(16, 30))
for estimator in (EmpiricalCovariance(), LedoitWolf()):
    estimator.fit(X)
    print(type(estimator).__name__,
          "rank=", np.linalg.matrix_rank(estimator.covariance_),
          "condition=", round(np.linalg.cond(estimator.covariance_), 2))
