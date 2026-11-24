#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تحليل التمييز التربيعي
Mastering scikit-learn 1.9 — Chapter 57 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-57-ch57 (lesson 57)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-57-ch57.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تحليل التمييز التربيعي
Mastering scikit-learn 1.9 — Chapter 57 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-57-ch57 (lesson 57)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# فئة تحتوي عنقودين داخليين: بنية توليدية غير متجانسة
X, y = make_classification(
    n_samples=1200, n_features=6, n_informative=6, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.5, random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    "LogisticRegression (تمييزي)": LogisticRegression(max_iter=1000),
    "LDA (توليدي، تغاير مشترك)":   LinearDiscriminantAnalysis(),
    "QDA (توليدي، تغاير لكل فئة)": QuadraticDiscriminantAnalysis(),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    print(f"{name:32s} دقة الاختبار: {model.score(X_te, y_te):.4f}")

lda = models["LDA (توليدي، تغاير مشترك)"]
print(f"ما تعلمه LDA توليدياً: priors_ = {lda.priors_}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# فئتان غاوسيتان بتغاير مشترك جيد الإشراط
rng = np.random.default_rng(42)
cov = np.array([[2.0, 0.4], [0.4, 1.0]])
X0 = rng.multivariate_normal([-1.5, -1.0], cov, size=400)
X1 = rng.multivariate_normal([1.5, 1.0], cov, size=400)
X = np.vstack([X0, X1])
y = np.r_[np.zeros(400), np.ones(400)]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LinearDiscriminantAnalysis(solver="lsqr").fit(X_tr, y_tr)
Sigma_inv = np.linalg.inv(model.covariance_)
mu, ln_pi = model.means_, np.log(model.priors_)
s0 = X_te @ (Sigma_inv @ mu[0]) - 0.5 * (mu[0] @ Sigma_inv @ mu[0]) + ln_pi[0]
s1 = X_te @ (Sigma_inv @ mu[1]) - 0.5 * (mu[1] @ Sigma_inv @ mu[1]) + ln_pi[1]

print(f"أقصى فرق بين درجتنا اليدوية و decision_function: {np.abs((s1 - s0) - model.decision_function(X_te)).max():.2e}")
print(f"توافق التصنيف اليدوي مع predict: {((s1 > s0) == (model.predict(X_te) == 1)).mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def battle(X, y, label):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    a = LinearDiscriminantAnalysis().fit(X_tr, y_tr).score(X_te, y_te)
    b = QuadraticDiscriminantAnalysis().fit(X_tr, y_tr).score(X_te, y_te)
    print(f"{label:42s} LDA = {a:.4f} | QDA = {b:.4f}")

# سيناريو 1: تغاير متماثل وعينات وفيرة — LDA هو النموذج الصحيح
rng = np.random.default_rng(42)
cov_same = np.diag([2.0] * 8)
mu = 1.4 / np.sqrt(8)
X0 = rng.multivariate_normal([-mu] * 8, cov_same, size=300)
X1 = rng.multivariate_normal([mu] * 8, cov_same, size=300)
battle(np.vstack([X0, X1]), np.r_[np.zeros(300), np.ones(300)], "تغاير متماثل + 300 عينة/فئة")

# سيناريو 2: التوزيع نفسه لكن 15 عينة فقط لكل فئة
rng = np.random.default_rng(7)
X0 = rng.multivariate_normal([-mu] * 8, cov_same, size=15)
X1 = rng.multivariate_normal([mu] * 8, cov_same, size=15)
battle(np.vstack([X0, X1]), np.r_[np.zeros(15), np.ones(15)], "تغاير متماثل + 15 عينة/فئة")

# سيناريو 3: فئات بأشكال إهليلجية مختلفة تماماً
rng = np.random.default_rng(42)
X0 = rng.multivariate_normal([-2, -2], np.diag([9.0, 1.0]), size=600)
X1 = rng.multivariate_normal([2, 2], np.diag([1.0, 9.0]), size=600)
battle(np.vstack([X0, X1]), np.r_[np.zeros(600), np.ones(600)], "إهليجات مختلفة + 600 عينة/فئة")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
Z = lda.transform(X)

print(f"الأبعاد الأصلية: {X.shape[1]} → بعد LDA: {Z.shape[1]}")
print(f"نسبة التباين المفسَّر لكل مكون: {lda.explained_variance_ratio_.round(4)}")
print(f"دقة اللوجيستي على المكونين: {LogisticRegression(max_iter=1000).fit(Z, y).score(Z, y):.4f}")
print(f"دقة اللوجيستي على الأبعاد الأربع: {LogisticRegression(max_iter=1000).fit(X, y).score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# 80 ميزة مقابل 70 عينة تدريب فقط
X, y = make_classification(n_samples=100, n_features=80, n_informative=30,
                           n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"أبعاد التدريب: {X_tr.shape} — نقدّر تغايراً 80×80 من 70 عينة!\n")

configs = [
    ("svd بلا انكماش",        LinearDiscriminantAnalysis(solver="svd")),
    ("lsqr بلا انكماش",       LinearDiscriminantAnalysis(solver="lsqr")),
    ("lsqr shrinkage='auto'", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ("lsqr + LedoitWolf",     LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=LedoitWolf())),
]
for name, model in configs:
    model.fit(X_tr, y_tr)
    print(f"{name:26s} دقة الاختبار: {model.score(X_te, y_te):.4f}")

try:
    QuadraticDiscriminantAnalysis().fit(X_tr, y_tr)
except np.linalg.LinAlgError as err:
    print(f"\nQDA بلا حماية انهار: {str(err)[:70]}...")

q = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X_tr, y_tr)
print(f"QDA مع انكماش تلقائي: {q.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# فئة نادرة بحدة: 10% فقط من العينات
X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                           weights=[0.9, 0.1], flip_y=0.02, class_sep=1.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = [
    ("priors المتعلمة", LinearDiscriminantAnalysis()),
    ("priors=[0.5, 0.5]", LinearDiscriminantAnalysis(priors=[0.5, 0.5])),
]
for name, model in models:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    recall_rare = (pred[y_te == 1] == 1).mean()
    print(f"{name:20s} الدقة: {model.score(X_te, y_te):.4f} | "
          f"الدقة المتوازنة: {balanced_accuracy_score(y_te, pred):.4f} | "
          f"استرجاع الفئة النادرة: {recall_rare:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
models = [
    ("LDA", LinearDiscriminantAnalysis()),
    ("QDA(reg_param=0.1)", QuadraticDiscriminantAnalysis(reg_param=0.1)),
]
for name, model in models:
    scores = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=5)
    print(f"{name:22s} متوسط الدقة الخماسية: {scores.mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch57.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=80, n_informative=30,
                           n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
for sh in [None, 0.1, 0.3, 0.5, 0.9, "auto"]:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=sh).fit(X_tr, y_tr)
    print(f"shrinkage={str(sh):6s} دقة الاختبار: {model.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# فئة تحتوي عنقودين داخليين: بنية توليدية غير متجانسة
X, y = make_classification(
    n_samples=1200, n_features=6, n_informative=6, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.5, random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    "LogisticRegression (تمييزي)": LogisticRegression(max_iter=1000),
    "LDA (توليدي، تغاير مشترك)":   LinearDiscriminantAnalysis(),
    "QDA (توليدي، تغاير لكل فئة)": QuadraticDiscriminantAnalysis(),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    print(f"{name:32s} دقة الاختبار: {model.score(X_te, y_te):.4f}")

lda = models["LDA (توليدي، تغاير مشترك)"]
print(f"ما تعلمه LDA توليدياً: priors_ = {lda.priors_}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# فئتان غاوسيتان بتغاير مشترك جيد الإشراط
rng = np.random.default_rng(42)
cov = np.array([[2.0, 0.4], [0.4, 1.0]])
X0 = rng.multivariate_normal([-1.5, -1.0], cov, size=400)
X1 = rng.multivariate_normal([1.5, 1.0], cov, size=400)
X = np.vstack([X0, X1])
y = np.r_[np.zeros(400), np.ones(400)]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LinearDiscriminantAnalysis(solver="lsqr").fit(X_tr, y_tr)
Sigma_inv = np.linalg.inv(model.covariance_)
mu, ln_pi = model.means_, np.log(model.priors_)
s0 = X_te @ (Sigma_inv @ mu[0]) - 0.5 * (mu[0] @ Sigma_inv @ mu[0]) + ln_pi[0]
s1 = X_te @ (Sigma_inv @ mu[1]) - 0.5 * (mu[1] @ Sigma_inv @ mu[1]) + ln_pi[1]

print(f"أقصى فرق بين درجتنا اليدوية و decision_function: {np.abs((s1 - s0) - model.decision_function(X_te)).max():.2e}")
print(f"توافق التصنيف اليدوي مع predict: {((s1 > s0) == (model.predict(X_te) == 1)).mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def battle(X, y, label):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    a = LinearDiscriminantAnalysis().fit(X_tr, y_tr).score(X_te, y_te)
    b = QuadraticDiscriminantAnalysis().fit(X_tr, y_tr).score(X_te, y_te)
    print(f"{label:42s} LDA = {a:.4f} | QDA = {b:.4f}")

# سيناريو 1: تغاير متماثل وعينات وفيرة — LDA هو النموذج الصحيح
rng = np.random.default_rng(42)
cov_same = np.diag([2.0] * 8)
mu = 1.4 / np.sqrt(8)
X0 = rng.multivariate_normal([-mu] * 8, cov_same, size=300)
X1 = rng.multivariate_normal([mu] * 8, cov_same, size=300)
battle(np.vstack([X0, X1]), np.r_[np.zeros(300), np.ones(300)], "تغاير متماثل + 300 عينة/فئة")

# سيناريو 2: التوزيع نفسه لكن 15 عينة فقط لكل فئة
rng = np.random.default_rng(7)
X0 = rng.multivariate_normal([-mu] * 8, cov_same, size=15)
X1 = rng.multivariate_normal([mu] * 8, cov_same, size=15)
battle(np.vstack([X0, X1]), np.r_[np.zeros(15), np.ones(15)], "تغاير متماثل + 15 عينة/فئة")

# سيناريو 3: فئات بأشكال إهليلجية مختلفة تماماً
rng = np.random.default_rng(42)
X0 = rng.multivariate_normal([-2, -2], np.diag([9.0, 1.0]), size=600)
X1 = rng.multivariate_normal([2, 2], np.diag([1.0, 9.0]), size=600)
battle(np.vstack([X0, X1]), np.r_[np.zeros(600), np.ones(600)], "إهليجات مختلفة + 600 عينة/فئة")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
Z = lda.transform(X)

print(f"الأبعاد الأصلية: {X.shape[1]} → بعد LDA: {Z.shape[1]}")
print(f"نسبة التباين المفسَّر لكل مكون: {lda.explained_variance_ratio_.round(4)}")
print(f"دقة اللوجيستي على المكونين: {LogisticRegression(max_iter=1000).fit(Z, y).score(Z, y):.4f}")
print(f"دقة اللوجيستي على الأبعاد الأربع: {LogisticRegression(max_iter=1000).fit(X, y).score(X, y):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# 80 ميزة مقابل 70 عينة تدريب فقط
X, y = make_classification(n_samples=100, n_features=80, n_informative=30,
                           n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"أبعاد التدريب: {X_tr.shape} — نقدّر تغايراً 80×80 من 70 عينة!\n")

configs = [
    ("svd بلا انكماش",        LinearDiscriminantAnalysis(solver="svd")),
    ("lsqr بلا انكماش",       LinearDiscriminantAnalysis(solver="lsqr")),
    ("lsqr shrinkage='auto'", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ("lsqr + LedoitWolf",     LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=LedoitWolf())),
]
for name, model in configs:
    model.fit(X_tr, y_tr)
    print(f"{name:26s} دقة الاختبار: {model.score(X_te, y_te):.4f}")

try:
    QuadraticDiscriminantAnalysis().fit(X_tr, y_tr)
except np.linalg.LinAlgError as err:
    print(f"\nQDA بلا حماية انهار: {str(err)[:70]}...")

q = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X_tr, y_tr)
print(f"QDA مع انكماش تلقائي: {q.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# فئة نادرة بحدة: 10% فقط من العينات
X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                           weights=[0.9, 0.1], flip_y=0.02, class_sep=1.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = [
    ("priors المتعلمة", LinearDiscriminantAnalysis()),
    ("priors=[0.5, 0.5]", LinearDiscriminantAnalysis(priors=[0.5, 0.5])),
]
for name, model in models:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    recall_rare = (pred[y_te == 1] == 1).mean()
    print(f"{name:20s} الدقة: {model.score(X_te, y_te):.4f} | "
          f"الدقة المتوازنة: {balanced_accuracy_score(y_te, pred):.4f} | "
          f"استرجاع الفئة النادرة: {recall_rare:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
models = [
    ("LDA", LinearDiscriminantAnalysis()),
    ("QDA(reg_param=0.1)", QuadraticDiscriminantAnalysis(reg_param=0.1)),
]
for name, model in models:
    scores = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=5)
    print(f"{name:22s} متوسط الدقة الخماسية: {scores.mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-57-ch57.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=80, n_informative=30,
                           n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
for sh in [None, 0.1, 0.3, 0.5, 0.9, "auto"]:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=sh).fit(X_tr, y_tr)
    print(f"shrinkage={str(sh):6s} دقة الاختبار: {model.score(X_te, y_te):.4f}")
