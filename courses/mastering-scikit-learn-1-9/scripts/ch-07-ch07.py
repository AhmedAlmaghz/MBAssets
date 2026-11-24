#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — فحص مُقدِّر قبل أن يذهب إلى الإنتاج
Mastering scikit-learn 1.9 — Chapter 07 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-07-ch07 (lesson 7)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-07-ch07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — فحص مُقدِّر قبل أن يذهب إلى الإنتاج
Mastering scikit-learn 1.9 — Chapter 07 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-07-ch07 (lesson 7)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import estimator_checks_generator

clf = DecisionTreeClassifier(random_state=42)
checks = list(estimator_checks_generator(clf))

print(f"عدد الفحوص الإجمالي للمصنّف: {len(checks)}")
print("\nعيّنة من أول ستة فحوص يمر بها النموذج:")
for est, check in checks[:6]:
    fn_name = check.func.__name__ if hasattr(check, "func") else getattr(check, "__name__", str(check))
    print(f"  - {fn_name}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init

class FaultyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # خطأ فادح: وضع سمة تدريب أو سمة إضافية تنتهي بشرطة داخل الباني!
        self.is_ready_ = False

est = FaultyClassifier()
try:
    check_no_attributes_set_in_init("FaultyClassifier", est)
except AssertionError as exc:
    print("نجح الفاحص في التقاط المخالفة:")
    print(str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

lr = LogisticRegression()
tags_lr = lr.__sklearn_tags__()

scaler = StandardScaler()
tags_scaler = scaler.__sklearn_tags__()

print("نوع كائن الوسوم:", type(tags_lr).__name__)
print("وسوم المصنّف اللوجيستي:")
print("  - يتطلب التدريب (requires_fit):", tags_lr.requires_fit)
print("  - يدعم المصفوفات المتفرقة:", tags_lr.input_tags.sparse)
print("  - يقبل القيم الناقصة (allow_nan):", tags_lr.input_tags.allow_nan)
print("  - يدعم معالجة الرسوميات (array_api):", tags_lr.array_api_support)

print("\nوسوم المحوّل القياسي (StandardScaler):")
print("  - نوع المُقدِّر:", tags_scaler.estimator_type)
print("  - الحفاظ على الأنواع (preserves_dtype):", tags_scaler.transformer_tags.preserves_dtype)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class CustomSparseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, factor=2.0):
        self.factor = factor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # إعلان دعم البيانات المتفرقة بوضوح
        tags.input_tags.sparse = True
        # التصريح بأن المحوّل يحافظ على دقة الأرقام العشرية
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

transformer = CustomSparseTransformer()
custom_tags = transformer.__sklearn_tags__()

print("هل صرّح المحوّل المخصّص بدعم البيانات المتفرقة؟", custom_tags.input_tags.sparse)
print("قائمة الأنواع المحفوظة:", custom_tags.transformer_tags.preserves_dtype)

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import estimator_html_repr

# بناء خط أنابيب متسلسل
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.5, max_iter=250, random_state=42)
)

# استخراج التمثيل التفاعلي بصيغة HTML كاملة
html_code = estimator_html_repr(pipe)

print(f"حجم كود HTML التفاعلي المولّد: {len(html_code):,} حرف")
print("هل يتضمن بنية حاوية العرض المرئي (sk-container)؟", "sk-container" in html_code)
print("هل يحتوي على أسماء العناصر (StandardScaler و LogisticRegression)؟",
      "StandardScaler" in html_code and "LogisticRegression" in html_code)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
import sklearn
sklearn.set_config(display="diagram")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.estimator_checks import check_estimator

class OutlierClipper(TransformerMixin, BaseEstimator):
    """محوّل متوافق تماماً يقص القيم الشاذة عند حدود المئينات المحسوبة."""
    def __init__(self, lower_percentile=5.0, upper_percentile=95.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        # التحقق من صحة المصفوفة وتثبيت n_features_in_
        X = validate_data(self, X, reset=True)
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        # التأكد من أن النموذج دُرّب مسبقاً
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        # التحقق دون إعادة ضبط السمات
        X = validate_data(self, X, reset=False)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

# تسجيل نتائج الفحص الشامل
check_results = []
def record_status(**kwargs):
    check_results.append((kwargs["check_name"], kwargs["status"]))

clipper = OutlierClipper()
check_estimator(clipper, callback=record_status, on_fail=None)

status_summary = Counter(status for _, status in check_results)
failed_checks = [name for name, status in check_results if status == "failed"]

print(f"إجمالي الفحوص المنفّذة على OutlierClipper: {len(check_results)}")
print(f"ملخّص النتائج: {dict(status_summary)}")
print(f"قائمة الفحوص الفاشلة: {failed_checks}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part01/ch07.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class ZeroMeanScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_"])
        X = validate_data(self, X, reset=False)
        return X - self.mean_

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import estimator_checks_generator

clf = DecisionTreeClassifier(random_state=42)
checks = list(estimator_checks_generator(clf))

print(f"عدد الفحوص الإجمالي للمصنّف: {len(checks)}")
print("\nعيّنة من أول ستة فحوص يمر بها النموذج:")
for est, check in checks[:6]:
    fn_name = check.func.__name__ if hasattr(check, "func") else getattr(check, "__name__", str(check))
    print(f"  - {fn_name}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init

class FaultyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # خطأ فادح: وضع سمة تدريب أو سمة إضافية تنتهي بشرطة داخل الباني!
        self.is_ready_ = False

est = FaultyClassifier()
try:
    check_no_attributes_set_in_init("FaultyClassifier", est)
except AssertionError as exc:
    print("نجح الفاحص في التقاط المخالفة:")
    print(str(exc).splitlines()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

lr = LogisticRegression()
tags_lr = lr.__sklearn_tags__()

scaler = StandardScaler()
tags_scaler = scaler.__sklearn_tags__()

print("نوع كائن الوسوم:", type(tags_lr).__name__)
print("وسوم المصنّف اللوجيستي:")
print("  - يتطلب التدريب (requires_fit):", tags_lr.requires_fit)
print("  - يدعم المصفوفات المتفرقة:", tags_lr.input_tags.sparse)
print("  - يقبل القيم الناقصة (allow_nan):", tags_lr.input_tags.allow_nan)
print("  - يدعم معالجة الرسوميات (array_api):", tags_lr.array_api_support)

print("\nوسوم المحوّل القياسي (StandardScaler):")
print("  - نوع المُقدِّر:", tags_scaler.estimator_type)
print("  - الحفاظ على الأنواع (preserves_dtype):", tags_scaler.transformer_tags.preserves_dtype)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class CustomSparseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, factor=2.0):
        self.factor = factor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # إعلان دعم البيانات المتفرقة بوضوح
        tags.input_tags.sparse = True
        # التصريح بأن المحوّل يحافظ على دقة الأرقام العشرية
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

transformer = CustomSparseTransformer()
custom_tags = transformer.__sklearn_tags__()

print("هل صرّح المحوّل المخصّص بدعم البيانات المتفرقة؟", custom_tags.input_tags.sparse)
print("قائمة الأنواع المحفوظة:", custom_tags.transformer_tags.preserves_dtype)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import estimator_html_repr

# بناء خط أنابيب متسلسل
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.5, max_iter=250, random_state=42)
)

# استخراج التمثيل التفاعلي بصيغة HTML كاملة
html_code = estimator_html_repr(pipe)

print(f"حجم كود HTML التفاعلي المولّد: {len(html_code):,} حرف")
print("هل يتضمن بنية حاوية العرض المرئي (sk-container)؟", "sk-container" in html_code)
print("هل يحتوي على أسماء العناصر (StandardScaler و LogisticRegression)؟",
      "StandardScaler" in html_code and "LogisticRegression" in html_code)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import sklearn
sklearn.set_config(display="diagram")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.estimator_checks import check_estimator

class OutlierClipper(TransformerMixin, BaseEstimator):
    """محوّل متوافق تماماً يقص القيم الشاذة عند حدود المئينات المحسوبة."""
    def __init__(self, lower_percentile=5.0, upper_percentile=95.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        # التحقق من صحة المصفوفة وتثبيت n_features_in_
        X = validate_data(self, X, reset=True)
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        # التأكد من أن النموذج دُرّب مسبقاً
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        # التحقق دون إعادة ضبط السمات
        X = validate_data(self, X, reset=False)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

# تسجيل نتائج الفحص الشامل
check_results = []
def record_status(**kwargs):
    check_results.append((kwargs["check_name"], kwargs["status"]))

clipper = OutlierClipper()
check_estimator(clipper, callback=record_status, on_fail=None)

status_summary = Counter(status for _, status in check_results)
failed_checks = [name for name, status in check_results if status == "failed"]

print(f"إجمالي الفحوص المنفّذة على OutlierClipper: {len(check_results)}")
print(f"ملخّص النتائج: {dict(status_summary)}")
print(f"قائمة الفحوص الفاشلة: {failed_checks}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-07-ch07.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class ZeroMeanScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_"])
        X = validate_data(self, X, reset=False)
        return X - self.mean_
