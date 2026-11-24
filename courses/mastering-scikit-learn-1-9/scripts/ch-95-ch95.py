#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كتابة مُقدِّر متوافق من الصفر
Mastering scikit-learn 1.9 — Chapter 95 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-95-ch95 (lesson 95)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-95-ch95.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — كتابة مُقدِّر متوافق من الصفر
Mastering scikit-learn 1.9 — Chapter 95 — Advanced

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-95-ch95 (lesson 95)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
import warnings
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

X, y = make_classification(n_samples=360, n_features=8, n_informative=5,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           weights=[0.50, 0.30, 0.20], random_state=42)


class MajorityBaseline(ClassifierMixin, BaseEstimator):
    """خط أساس: يتنبّأ بالفئة التي حصّتها في التدريب الأقرب إلى `threshold`."""

    _parameter_constraints = {
        "threshold": [Interval(type=Real, left=0.0, right=1.0, closed="both")],
    }

    def __init__(self, threshold=0.5):
        self.threshold = threshold          # تخزين حرفي: لا شرط ولا تحويل

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes             # سمات متعلَّمة: بشرطة سفلية
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)        # تحقّق من الحالة قبل قراءة السمات
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.classifier_tags.poor_score = True
        return tags


model = MajorityBaseline()
print("get_params:", model.get_params())
print("محتوى الكائن:", model.__dict__)
print("قبل fit — classes_:", hasattr(model, "classes_"),
      "| frequencies_:", hasattr(model, "frequencies_"))

copy = clone(model)
print("clone أعطى كائناً جديداً:", copy is not model,
      "| بنفس البارامترات:", copy.get_params() == model.get_params())
print("نسخة clone غير مدرَّبة:", hasattr(copy, "classes_"))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
try:
    MajorityBaseline().predict(X)
except NotFittedError as exc:
    print("NotFittedError:", exc)

model.fit(X, y)
print("سمات بعد fit:", sorted(k for k in vars(model) if k.endswith("_")))
print("الحصص:", np.round(model.frequencies_, 4))
print("التنبؤ:", np.unique(model.predict(X)), "| الدرجة:", round(model.score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
fitted = MajorityBaseline().fit(X, y)
print("عدد الخصائص المسجَّل:", fitted.n_features_in_)
print("أسماء الخصائص:", list(getattr(fitted, "feature_names_in_", []))[:3] or "غير موجودة")

try:
    fitted.predict(X[:, :3])
except ValueError as exc:
    print("ValueError:", exc)

X_bad = X.copy()
X_bad[0, 0] = np.nan
try:
    MajorityBaseline().fit(X_bad, y)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0])

try:
    MajorityBaseline().fit(X, y[:-1])
except ValueError as exc:
    print("ValueError:", exc)

try:
    MajorityBaseline().fit(X, np.linspace(0.0, 1.0, X.shape[0]))
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.estimator_checks import check_estimator


class WrongOrder(BaseEstimator, ClassifierMixin):
    """نفس الشيء تماماً، لكن BaseEstimator يقف على اليسار."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        check_estimator(WrongOrder())
    except AttributeError as exc:
        print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils._tags import InputTags, Tags, TargetTags, get_tags

lr_tags = get_tags(LogisticRegression())
print("حقول Tags:", list(Tags.__dataclass_fields__))
print("حقول InputTags:", list(InputTags.__dataclass_fields__))
print("حقول TargetTags:", list(TargetTags.__dataclass_fields__))
print()
print("الانحدار اللوجيستي — النوع:", lr_tags.estimator_type,
      "| يحتاج fit:", lr_tags.requires_fit)
print("  المدخلات: ثنائية الأبعاد:", lr_tags.input_tags.two_d_array,
      "| متفرّقة:", lr_tags.input_tags.sparse,
      "| تسمح بالمفقود:", lr_tags.input_tags.allow_nan)
print("  الهدف: مطلوب:", lr_tags.target_tags.required,
      "| متعدّد المخرجات:", lr_tags.target_tags.multi_output)
print("  المصنِّف: متعدّد الأصناف:", lr_tags.classifier_tags.multi_class,
      "| متعدّد التسميات:", lr_tags.classifier_tags.multi_label,
      "| أداء ضعيف:", lr_tags.classifier_tags.poor_score)

sc_tags = get_tags(StandardScaler())
print("المُطبِّع المعياري — النوع:", sc_tags.estimator_type,
      "| يحتاج fit:", sc_tags.requires_fit)
print("  المدخلات: ثنائية الأبعاد:", sc_tags.input_tags.two_d_array,
      "| متفرّقة:", sc_tags.input_tags.sparse,
      "| تسمح بالمفقود:", sc_tags.input_tags.allow_nan)
print("  الهدف: مطلوب:", sc_tags.target_tags.required)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
own = get_tags(MajorityBaseline())
print("نوعنا:", own.estimator_type, "| يحتاج fit:", own.requires_fit)
print("مدخلاتنا: ثنائية:", own.input_tags.two_d_array,
      "| متفرّقة:", own.input_tags.sparse,
      "| قيم مفقودة:", own.input_tags.allow_nan)
print("هدفنا: مطلوب:", own.target_tags.required,
      "| متعدّد المخرجات:", own.target_tags.multi_output)
print("صنّافنا: أداء ضعيف:", own.classifier_tags.poor_score)

# ────────────────────────────────────────────────────────────────────────
# [007] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier


class HonestBaseline(ClassifierMixin, BaseEstimator):
    """خط أساس مدرَّب، مثل صنفنا لكنه يعلن ضعفه."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


weak = HonestBaseline().fit(X, y)
strong = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print("خط الأساس    :", round(weak.score(X, y), 4),
      "| وسم poor_score:", get_tags(weak).classifier_tags.poor_score)
print("شجرة القرار  :", round(strong.score(X, y), 4),
      "| وسم poor_score:", get_tags(strong).classifier_tags.poor_score)

# ────────────────────────────────────────────────────────────────────────
# [008] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
class NaNAware(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True      # مصرّح به في الوسم
        return tags

# ────────────────────────────────────────────────────────────────────────
# [009] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

lr_constraints = LogisticRegression()._parameter_constraints
print("C            :", type(lr_constraints["C"][0]).__name__, "→", lr_constraints["C"][0])
print("fit_intercept:", lr_constraints["fit_intercept"])
print("solver       :", type(lr_constraints["solver"][0]).__name__,
      "→", sorted(lr_constraints["solver"][0].options))
print("class_weight :", lr_constraints["class_weight"])
print("قيدنا        :", MajorityBaseline._parameter_constraints["threshold"][0])

# ────────────────────────────────────────────────────────────────────────
# [010] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
loose = MajorityBaseline()
loose.set_params(threshold=3.0)
print("set_params قبلت 3.0:", loose.get_params())
try:
    loose.fit(X, y)
except Exception as exc:
    print(type(exc).__name__, ":", exc)

good = MajorityBaseline().set_params(threshold=0.35)
print("قيمة صالحة:", good.get_params()["threshold"],
      "| الدرجة:", round(good.fit(X, y).score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [011] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("sc", StandardScaler()),
                 ("clf", MajorityBaseline())])
print("بارامترات الأنبوب:", sorted(pipe.get_params(deep=False)))
print("عنوان مُتداخل يعمل:", "clf__threshold" in pipe.get_params(deep=True))

grid = GridSearchCV(pipe, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid.fit(X, y)
print("أفضل تركيبة:", grid.best_params_)
print("درجات الشبكة:", np.round(grid.cv_results_["mean_test_score"], 4))
print("درجة التحقق:", round(float(grid.best_score_), 4))

folds = cross_val_score(pipe, X, y, cv=5)
print("التحقق المتقاطع:", np.round(folds, 4),
      "| المتوسط:", round(float(folds.mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [012] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        check_estimator(MajorityBaseline())
        print("النتيجة: اجتاز كل الفحوص")
    except Exception as exc:
        print("النتيجة:", type(exc).__name__, "| الرسالة:", repr(str(exc)))

skipped = [w for w in caught if w.category.__name__ == "SkipTestWarning"]
print("عدد التخطّيات:", len(skipped))
for w in skipped:
    print("  ", str(w.message)[:108])

# ────────────────────────────────────────────────────────────────────────
# [013] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnShifter(TransformerMixin, BaseEstimator):
    """محرّك بسيط: ينسخ كل عمود ويُنقص منه متوسطه."""

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="numeric", ensure_all_finite=True)
        self.n_features_in_ = X.shape[1]
        self.offsets_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        return X - self.offsets_


shift = ColumnShifter()
print("يملك transform     :", hasattr(shift, "transform"))
print("يملك fit_transform :", hasattr(shift, "fit_transform"))
manual = shift.fit(X).transform(X)
fast = ColumnShifter().fit_transform(X)
print("fit_transform = fit ثم transform:", bool(np.allclose(manual, fast)))

shifted = Pipeline([("shift", ColumnShifter()),
                    ("clf", MajorityBaseline())]).fit(X, y)
print("داخل الأنبوب:", round(shifted.score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [014] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
class NearestCentroidFromScratch(ClassifierMixin, BaseEstimator):
    """خط أساس بسيط: يتنبّأ بالفئة التي حصّتها الأقرب إلى `shrinkage`."""

    _parameter_constraints = {
        "shrinkage": [Interval(type=Real, left=0.0, right=1.0, closed="both")],
    }

    def __init__(self, shrinkage=0.0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        target = self.classes_[np.argmin(np.abs(self.frequencies_
                                                 - float(self.shrinkage)))]
        return np.full(X.shape[0], target)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


scratch = NearestCentroidFromScratch()
print("get_params:", scratch.get_params())
print("محتوى الكائن قبل fit:", scratch.__dict__)
try:
    scratch.predict(X)
except NotFittedError as exc:
    print("NotFittedError:", str(exc)[:58], "…")

scratch.fit(X, y)
print("سمات بعد fit:", sorted(k for k in vars(scratch) if k.endswith("_")))
try:
    scratch.predict(X[:, :3])
except ValueError as exc:
    print("ValueError:", exc)

try:
    NearestCentroidFromScratch(shrinkage=1.5).fit(X, y)
except Exception as exc:
    print(type(exc).__name__, ":", exc)

# ────────────────────────────────────────────────────────────────────────
# [015] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
class BadOrder(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


class BadNaN(BadOrder):
    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric",
                             ensure_all_finite="allow-nan")
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self


for name, est in (("BadOrder", BadOrder()), ("BadNaN", BadNaN())):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check_estimator(est)
            print(name, ": اجتاز")
        except Exception as exc:
            print(name, "→", type(exc).__name__, "|", str(exc).splitlines()[0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        check_estimator(MajorityBaseline())
        print("بعد الإصلاح: اجتاز")
    except Exception as exc:
        print("بعد الإصلاح:", type(exc).__name__, "|", str(exc))

# ────────────────────────────────────────────────────────────────────────
# [016] content/part11/ch95.md
# ────────────────────────────────────────────────────────────────────────
pipe_two = Pipeline([("sc", StandardScaler()),
                     ("clf", MajorityBaseline())])
grid_two = GridSearchCV(pipe_two, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid_two.fit(X, y)
print("خطوتان  →", grid_two.best_params_, "| الدرجات:",
      np.round(grid_two.cv_results_["mean_test_score"], 4))

pipe_three = Pipeline([("shift", ColumnShifter()),
                       ("sc", StandardScaler()),
                       ("clf", MajorityBaseline())])
grid_three = GridSearchCV(pipe_three, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid_three.fit(X, y)
print("ثلاث خطوات →", grid_three.best_params_, "| الدرجات:",
      np.round(grid_three.cv_results_["mean_test_score"], 4))

print("الفارق:", np.round(grid_two.cv_results_["mean_test_score"]
                          - grid_three.cv_results_["mean_test_score"], 4))
print("ثلاث خطوات عبر تحقق متقاطع:",
      np.round(cross_val_score(pipe_three, X, y, cv=5), 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import warnings
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

X, y = make_classification(n_samples=360, n_features=8, n_informative=5,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           weights=[0.50, 0.30, 0.20], random_state=42)


class MajorityBaseline(ClassifierMixin, BaseEstimator):
    """خط أساس: يتنبّأ بالفئة التي حصّتها في التدريب الأقرب إلى `threshold`."""

    _parameter_constraints = {
        "threshold": [Interval(type=Real, left=0.0, right=1.0, closed="both")],
    }

    def __init__(self, threshold=0.5):
        self.threshold = threshold          # تخزين حرفي: لا شرط ولا تحويل

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes             # سمات متعلَّمة: بشرطة سفلية
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)        # تحقّق من الحالة قبل قراءة السمات
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.classifier_tags.poor_score = True
        return tags


model = MajorityBaseline()
print("get_params:", model.get_params())
print("محتوى الكائن:", model.__dict__)
print("قبل fit — classes_:", hasattr(model, "classes_"),
      "| frequencies_:", hasattr(model, "frequencies_"))

copy = clone(model)
print("clone أعطى كائناً جديداً:", copy is not model,
      "| بنفس البارامترات:", copy.get_params() == model.get_params())
print("نسخة clone غير مدرَّبة:", hasattr(copy, "classes_"))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
try:
    MajorityBaseline().predict(X)
except NotFittedError as exc:
    print("NotFittedError:", exc)

model.fit(X, y)
print("سمات بعد fit:", sorted(k for k in vars(model) if k.endswith("_")))
print("الحصص:", np.round(model.frequencies_, 4))
print("التنبؤ:", np.unique(model.predict(X)), "| الدرجة:", round(model.score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
fitted = MajorityBaseline().fit(X, y)
print("عدد الخصائص المسجَّل:", fitted.n_features_in_)
print("أسماء الخصائص:", list(getattr(fitted, "feature_names_in_", []))[:3] or "غير موجودة")

try:
    fitted.predict(X[:, :3])
except ValueError as exc:
    print("ValueError:", exc)

X_bad = X.copy()
X_bad[0, 0] = np.nan
try:
    MajorityBaseline().fit(X_bad, y)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0])

try:
    MajorityBaseline().fit(X, y[:-1])
except ValueError as exc:
    print("ValueError:", exc)

try:
    MajorityBaseline().fit(X, np.linspace(0.0, 1.0, X.shape[0]))
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.utils.estimator_checks import check_estimator


class WrongOrder(BaseEstimator, ClassifierMixin):
    """نفس الشيء تماماً، لكن BaseEstimator يقف على اليسار."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        check_estimator(WrongOrder())
    except AttributeError as exc:
        print(f"{type(exc).__name__}: {exc}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils._tags import InputTags, Tags, TargetTags, get_tags

lr_tags = get_tags(LogisticRegression())
print("حقول Tags:", list(Tags.__dataclass_fields__))
print("حقول InputTags:", list(InputTags.__dataclass_fields__))
print("حقول TargetTags:", list(TargetTags.__dataclass_fields__))
print()
print("الانحدار اللوجيستي — النوع:", lr_tags.estimator_type,
      "| يحتاج fit:", lr_tags.requires_fit)
print("  المدخلات: ثنائية الأبعاد:", lr_tags.input_tags.two_d_array,
      "| متفرّقة:", lr_tags.input_tags.sparse,
      "| تسمح بالمفقود:", lr_tags.input_tags.allow_nan)
print("  الهدف: مطلوب:", lr_tags.target_tags.required,
      "| متعدّد المخرجات:", lr_tags.target_tags.multi_output)
print("  المصنِّف: متعدّد الأصناف:", lr_tags.classifier_tags.multi_class,
      "| متعدّد التسميات:", lr_tags.classifier_tags.multi_label,
      "| أداء ضعيف:", lr_tags.classifier_tags.poor_score)

sc_tags = get_tags(StandardScaler())
print("المُطبِّع المعياري — النوع:", sc_tags.estimator_type,
      "| يحتاج fit:", sc_tags.requires_fit)
print("  المدخلات: ثنائية الأبعاد:", sc_tags.input_tags.two_d_array,
      "| متفرّقة:", sc_tags.input_tags.sparse,
      "| تسمح بالمفقود:", sc_tags.input_tags.allow_nan)
print("  الهدف: مطلوب:", sc_tags.target_tags.required)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
own = get_tags(MajorityBaseline())
print("نوعنا:", own.estimator_type, "| يحتاج fit:", own.requires_fit)
print("مدخلاتنا: ثنائية:", own.input_tags.two_d_array,
      "| متفرّقة:", own.input_tags.sparse,
      "| قيم مفقودة:", own.input_tags.allow_nan)
print("هدفنا: مطلوب:", own.target_tags.required,
      "| متعدّد المخرجات:", own.target_tags.multi_output)
print("صنّافنا: أداء ضعيف:", own.classifier_tags.poor_score)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier


class HonestBaseline(ClassifierMixin, BaseEstimator):
    """خط أساس مدرَّب، مثل صنفنا لكنه يعلن ضعفه."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


weak = HonestBaseline().fit(X, y)
strong = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print("خط الأساس    :", round(weak.score(X, y), 4),
      "| وسم poor_score:", get_tags(weak).classifier_tags.poor_score)
print("شجرة القرار  :", round(strong.score(X, y), 4),
      "| وسم poor_score:", get_tags(strong).classifier_tags.poor_score)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class NaNAware(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True      # مصرّح به في الوسم
        return tags

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

lr_constraints = LogisticRegression()._parameter_constraints
print("C            :", type(lr_constraints["C"][0]).__name__, "→", lr_constraints["C"][0])
print("fit_intercept:", lr_constraints["fit_intercept"])
print("solver       :", type(lr_constraints["solver"][0]).__name__,
      "→", sorted(lr_constraints["solver"][0].options))
print("class_weight :", lr_constraints["class_weight"])
print("قيدنا        :", MajorityBaseline._parameter_constraints["threshold"][0])

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
loose = MajorityBaseline()
loose.set_params(threshold=3.0)
print("set_params قبلت 3.0:", loose.get_params())
try:
    loose.fit(X, y)
except Exception as exc:
    print(type(exc).__name__, ":", exc)

good = MajorityBaseline().set_params(threshold=0.35)
print("قيمة صالحة:", good.get_params()["threshold"],
      "| الدرجة:", round(good.fit(X, y).score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("sc", StandardScaler()),
                 ("clf", MajorityBaseline())])
print("بارامترات الأنبوب:", sorted(pipe.get_params(deep=False)))
print("عنوان مُتداخل يعمل:", "clf__threshold" in pipe.get_params(deep=True))

grid = GridSearchCV(pipe, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid.fit(X, y)
print("أفضل تركيبة:", grid.best_params_)
print("درجات الشبكة:", np.round(grid.cv_results_["mean_test_score"], 4))
print("درجة التحقق:", round(float(grid.best_score_), 4))

folds = cross_val_score(pipe, X, y, cv=5)
print("التحقق المتقاطع:", np.round(folds, 4),
      "| المتوسط:", round(float(folds.mean()), 4))

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        check_estimator(MajorityBaseline())
        print("النتيجة: اجتاز كل الفحوص")
    except Exception as exc:
        print("النتيجة:", type(exc).__name__, "| الرسالة:", repr(str(exc)))

skipped = [w for w in caught if w.category.__name__ == "SkipTestWarning"]
print("عدد التخطّيات:", len(skipped))
for w in skipped:
    print("  ", str(w.message)[:108])

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnShifter(TransformerMixin, BaseEstimator):
    """محرّك بسيط: ينسخ كل عمود ويُنقص منه متوسطه."""

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="numeric", ensure_all_finite=True)
        self.n_features_in_ = X.shape[1]
        self.offsets_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        return X - self.offsets_


shift = ColumnShifter()
print("يملك transform     :", hasattr(shift, "transform"))
print("يملك fit_transform :", hasattr(shift, "fit_transform"))
manual = shift.fit(X).transform(X)
fast = ColumnShifter().fit_transform(X)
print("fit_transform = fit ثم transform:", bool(np.allclose(manual, fast)))

shifted = Pipeline([("shift", ColumnShifter()),
                    ("clf", MajorityBaseline())]).fit(X, y)
print("داخل الأنبوب:", round(shifted.score(X, y), 4))

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
class NearestCentroidFromScratch(ClassifierMixin, BaseEstimator):
    """خط أساس بسيط: يتنبّأ بالفئة التي حصّتها الأقرب إلى `shrinkage`."""

    _parameter_constraints = {
        "shrinkage": [Interval(type=Real, left=0.0, right=1.0, closed="both")],
    }

    def __init__(self, shrinkage=0.0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        target = self.classes_[np.argmin(np.abs(self.frequencies_
                                                 - float(self.shrinkage)))]
        return np.full(X.shape[0], target)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


scratch = NearestCentroidFromScratch()
print("get_params:", scratch.get_params())
print("محتوى الكائن قبل fit:", scratch.__dict__)
try:
    scratch.predict(X)
except NotFittedError as exc:
    print("NotFittedError:", str(exc)[:58], "…")

scratch.fit(X, y)
print("سمات بعد fit:", sorted(k for k in vars(scratch) if k.endswith("_")))
try:
    scratch.predict(X[:, :3])
except ValueError as exc:
    print("ValueError:", exc)

try:
    NearestCentroidFromScratch(shrinkage=1.5).fit(X, y)
except Exception as exc:
    print(type(exc).__name__, ":", exc)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
class BadOrder(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric", ensure_all_finite=True)
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        scores = 1.0 / (1.0 + np.abs(self.frequencies_ - float(self.threshold)))
        return np.tile(scores / scores.sum(), (X.shape[0], 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


class BadNaN(BadOrder):
    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="numeric",
                             ensure_all_finite="allow-nan")
        check_classification_targets(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.frequencies_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        return self


for name, est in (("BadOrder", BadOrder()), ("BadNaN", BadNaN())):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check_estimator(est)
            print(name, ": اجتاز")
        except Exception as exc:
            print(name, "→", type(exc).__name__, "|", str(exc).splitlines()[0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        check_estimator(MajorityBaseline())
        print("بعد الإصلاح: اجتاز")
    except Exception as exc:
        print("بعد الإصلاح:", type(exc).__name__, "|", str(exc))

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-95-ch95.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
pipe_two = Pipeline([("sc", StandardScaler()),
                     ("clf", MajorityBaseline())])
grid_two = GridSearchCV(pipe_two, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid_two.fit(X, y)
print("خطوتان  →", grid_two.best_params_, "| الدرجات:",
      np.round(grid_two.cv_results_["mean_test_score"], 4))

pipe_three = Pipeline([("shift", ColumnShifter()),
                       ("sc", StandardScaler()),
                       ("clf", MajorityBaseline())])
grid_three = GridSearchCV(pipe_three, {"clf__threshold": [0.1, 0.35, 0.6]}, cv=5)
grid_three.fit(X, y)
print("ثلاث خطوات →", grid_three.best_params_, "| الدرجات:",
      np.round(grid_three.cv_results_["mean_test_score"], 4))

print("الفارق:", np.round(grid_two.cv_results_["mean_test_score"]
                          - grid_three.cv_results_["mean_test_score"], 4))
print("ثلاث خطوات عبر تحقق متقاطع:",
      np.round(cross_val_score(pipe_three, X, y, cv=5), 4))
