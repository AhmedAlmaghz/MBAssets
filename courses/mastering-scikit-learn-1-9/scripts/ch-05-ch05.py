#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نموذج التقدير: فهم Estimator API
Mastering scikit-learn 1.9 — Estimator

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-05-ch05 (lesson 5)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-05-ch05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — نموذج التقدير: فهم Estimator API
Mastering scikit-learn 1.9 — Estimator

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-05-ch05 (lesson 5)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
est = KNeighborsClassifier(n_neighbors=5)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

est = KNeighborsClassifier(n_neighbors=5)
est.predict(X)      # AttributeError: لا يوجد predict_proba بعد

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

est = KNeighborsClassifier(n_neighbors=5, weights="uniform")
print("قبل fit — ما الذي يعرفه الكائن؟")
print(repr(est))
print("السمات المنتهية بشرطة:", [a for a in vars(est) if a.endswith("_")])

est.fit(X, y)

print("\nبعد fit — ما الذي أضافه؟")
print("السمات المنتهية بشرطة:", sorted(a for a in vars(est) if a.endswith("_")))
print("n_features_in_ :", est.n_features_in_)
print("n_samples_fit_ :", est.n_samples_fit_)
print("classes_       :", est.classes_)
print("_fit_method    :", est._fit_method)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
est.set_params(n_neighbors=11)
est.fit(X[:100], y[:100])
print("بعد إعادة التدريب على مئة صف وصنفين فقط:")
print("  n_samples_fit_ :", est.n_samples_fit_, "  (كان ١٥٠)")
print("  classes_       :", est.classes_, "  (كان [0 1 2])")
print("  n_features_in_ :", est.n_features_in_, "  (بقي ٤ لأن الأعمدة نفسها)")
print("\nالسمات فوق أُعيد حسابها من الصفر. لا شيء متراكم من الطيّة السابقة.")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
fresh = KNeighborsClassifier()
print("كل المعاملات:", sorted(fresh.get_params().keys()))

fresh.set_params(n_neighbors=11, weights="distance")
print("\nبعد set_params:", fresh.get_params()["n_neighbors"],
      fresh.get_params()["weights"])
print("لم يتغيّر شيء آخر:", fresh.get_params()["metric"])

from sklearn.base import clone
copy_of = clone(fresh)
print("\nclone يعطي كائناً جديداً بنفس المعاملات:",
      copy_of.get_params()["n_neighbors"], "  وهو كائن آخر؟", copy_of is not fresh)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, get_scorer, make_scorer, recall_score

clf = LogisticRegression(max_iter=1000).fit(X, y)

print("clf.score(X, y)            :", round(clf.score(X, y), 4))
print("accuracy_score(y, predict) :", round(accuracy_score(y, clf.predict(X)), 4))
print("get_scorer('f1_macro')     :", round(get_scorer("f1_macro")(clf, X, y), 4))
print("make_scorer(recall, macro) :",
      round(make_scorer(recall_score, average="macro")(clf, X, y), 4))

print("\nالقاعدة: score() هي دقّة التصنيف للمصنِّف، ومعامل تحديد لـالانحدار.")
print("إن كانت مقياسك مختلفاً، مرّر السلسلة إلى cross_val_score أو GridSearchCV.")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

try:
    StandardScaler().fit_transform(np.array([1.0, 2.0, 3.0]))
except ValueError as exc:
    print("مصفوفة أحادية البعد:")
    print("  ", str(exc).splitlines()[0])

try:
    LogisticRegression().fit(X, None)
except ValueError as exc:
    print("\nهدف مفقود:")
    print("  ", exc)

knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
try:
    knn.predict(X[:, :3])
except ValueError as exc:
    print("\nعدد أعمدة خاطئ:")
    print("  ", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [008] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnMeanImputer(TransformerMixin, BaseEstimator):
    """Replace every missing value by the mean of its column."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        col = np.where(np.isnan(X), self.statistics_, X)
        return col


Xh = X.copy()
Xh[0, 1] = np.nan
Xh[5, 3] = np.nan

imp = ColumnMeanImputer().fit(Xh)
print("عدد الأعمدة تعلّمه   :", imp.n_features_in_)
print("متوسّط كل عمود       :", np.round(imp.statistics_, 4))
print("عدد القيم المملوءة   :", int(np.isnan(Xh).sum()))
print("ما زال مفقوداً       :", int(np.isnan(imp.transform(Xh)).sum()))
print("clone() يعيد كائناً جديداً فارغاً:",
      clone(imp).get_params() is not None,
      "| و transform قبل fit يرفع:",
      end=" ")
try:
    ColumnMeanImputer().transform(Xh)
except Exception as exc:
    print(type(exc).__name__)

# ────────────────────────────────────────────────────────────────────────
# [009] content/part01/ch05.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print("cross_val_score يتكفّل بالنسخ. النتائج:", np.round(cv, 4))

shared = LogisticRegression(max_iter=1000)
shared.fit(X[:100], y[:100])
fitted_after_first = hasattr(shared, "coef_")
shared.fit(X, y)
print("\nنفس الكائن تدرّب مرّتين.")
print("  كان مدرَّباً بعد الطيّة الأولى :", fitted_after_first)
print("  coef_ النهائي شكله            :", shared.coef_.shape)
print("  أيّ أصناف يراها الآن؟          :", len(set(y)))

folded = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
models = []
for tr, te in folded.split(X, y):
    m = clone(LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
    models.append(m.coef_.shape)
print("\nكل طيّة مع نسخة نظيفة، بأشكال:", models)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
est = KNeighborsClassifier(n_neighbors=5)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

est = KNeighborsClassifier(n_neighbors=5)
est.predict(X)      # AttributeError: لا يوجد predict_proba بعد

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

est = KNeighborsClassifier(n_neighbors=5, weights="uniform")
print("قبل fit — ما الذي يعرفه الكائن؟")
print(repr(est))
print("السمات المنتهية بشرطة:", [a for a in vars(est) if a.endswith("_")])

est.fit(X, y)

print("\nبعد fit — ما الذي أضافه؟")
print("السمات المنتهية بشرطة:", sorted(a for a in vars(est) if a.endswith("_")))
print("n_features_in_ :", est.n_features_in_)
print("n_samples_fit_ :", est.n_samples_fit_)
print("classes_       :", est.classes_)
print("_fit_method    :", est._fit_method)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
est.set_params(n_neighbors=11)
est.fit(X[:100], y[:100])
print("بعد إعادة التدريب على مئة صف وصنفين فقط:")
print("  n_samples_fit_ :", est.n_samples_fit_, "  (كان ١٥٠)")
print("  classes_       :", est.classes_, "  (كان [0 1 2])")
print("  n_features_in_ :", est.n_features_in_, "  (بقي ٤ لأن الأعمدة نفسها)")
print("\nالسمات فوق أُعيد حسابها من الصفر. لا شيء متراكم من الطيّة السابقة.")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
fresh = KNeighborsClassifier()
print("كل المعاملات:", sorted(fresh.get_params().keys()))

fresh.set_params(n_neighbors=11, weights="distance")
print("\nبعد set_params:", fresh.get_params()["n_neighbors"],
      fresh.get_params()["weights"])
print("لم يتغيّر شيء آخر:", fresh.get_params()["metric"])

from sklearn.base import clone
copy_of = clone(fresh)
print("\nclone يعطي كائناً جديداً بنفس المعاملات:",
      copy_of.get_params()["n_neighbors"], "  وهو كائن آخر؟", copy_of is not fresh)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, get_scorer, make_scorer, recall_score

clf = LogisticRegression(max_iter=1000).fit(X, y)

print("clf.score(X, y)            :", round(clf.score(X, y), 4))
print("accuracy_score(y, predict) :", round(accuracy_score(y, clf.predict(X)), 4))
print("get_scorer('f1_macro')     :", round(get_scorer("f1_macro")(clf, X, y), 4))
print("make_scorer(recall, macro) :",
      round(make_scorer(recall_score, average="macro")(clf, X, y), 4))

print("\nالقاعدة: score() هي دقّة التصنيف للمصنِّف، ومعامل تحديد لـالانحدار.")
print("إن كانت مقياسك مختلفاً، مرّر السلسلة إلى cross_val_score أو GridSearchCV.")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

try:
    StandardScaler().fit_transform(np.array([1.0, 2.0, 3.0]))
except ValueError as exc:
    print("مصفوفة أحادية البعد:")
    print("  ", str(exc).splitlines()[0])

try:
    LogisticRegression().fit(X, None)
except ValueError as exc:
    print("\nهدف مفقود:")
    print("  ", exc)

knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
try:
    knn.predict(X[:, :3])
except ValueError as exc:
    print("\nعدد أعمدة خاطئ:")
    print("  ", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnMeanImputer(TransformerMixin, BaseEstimator):
    """Replace every missing value by the mean of its column."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        col = np.where(np.isnan(X), self.statistics_, X)
        return col


Xh = X.copy()
Xh[0, 1] = np.nan
Xh[5, 3] = np.nan

imp = ColumnMeanImputer().fit(Xh)
print("عدد الأعمدة تعلّمه   :", imp.n_features_in_)
print("متوسّط كل عمود       :", np.round(imp.statistics_, 4))
print("عدد القيم المملوءة   :", int(np.isnan(Xh).sum()))
print("ما زال مفقوداً       :", int(np.isnan(imp.transform(Xh)).sum()))
print("clone() يعيد كائناً جديداً فارغاً:",
      clone(imp).get_params() is not None,
      "| و transform قبل fit يرفع:",
      end=" ")
try:
    ColumnMeanImputer().transform(Xh)
except Exception as exc:
    print(type(exc).__name__)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-05-ch05.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print("cross_val_score يتكفّل بالنسخ. النتائج:", np.round(cv, 4))

shared = LogisticRegression(max_iter=1000)
shared.fit(X[:100], y[:100])
fitted_after_first = hasattr(shared, "coef_")
shared.fit(X, y)
print("\nنفس الكائن تدرّب مرّتين.")
print("  كان مدرَّباً بعد الطيّة الأولى :", fitted_after_first)
print("  coef_ النهائي شكله            :", shared.coef_.shape)
print("  أيّ أصناف يراها الآن؟          :", len(set(y)))

folded = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
models = []
for tr, te in folded.split(X, y):
    m = clone(LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
    models.append(m.coef_.shape)
print("\nكل طيّة مع نسخة نظيفة، بأشكال:", models)
