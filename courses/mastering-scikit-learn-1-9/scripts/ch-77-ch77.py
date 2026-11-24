#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط عتبة القرار
Mastering scikit-learn 1.9 — Chapter 77 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-77-ch77 (lesson 77)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-77-ch77.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ضبط عتبة القرار
Mastering scikit-learn 1.9 — Chapter 77 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-77-ch77 (lesson 77)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

X, y = make_classification(
    n_samples=4000, n_features=10, n_informative=4,
    weights=[0.95, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
prob = model.predict_proba(X_te)[:, 1]

print("العتبة | الاسترجاع | الدقة | الدقة الخام")
for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    y_hat = (prob >= t).astype(int)
    rec = recall_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    acc = accuracy_score(y_te, y_hat)
    print(f"{t:6.1f} | {rec:9.4f} | {prec:5.4f} | {acc:11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_score, recall_score

frozen_model = FrozenEstimator(model)   # نجمّد النموذج: لا إعادة تدريب مع كل عتبة

print("العتبة | الاسترجاع | الدقة | الدقة الخام")
for t in [0.2, 0.3, 0.5, 0.7]:
    ftc = FixedThresholdClassifier(estimator=frozen_model, threshold=t)
    y_hat = ftc.predict(X_te)          # قطع عند العتبة المطلوبة مباشرة
    rec = recall_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    acc = accuracy_score(y_te, y_hat)
    print(f"{t:6.1f} | {rec:9.4f} | {prec:5.4f} | {acc:11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import TunedThresholdClassifierCV

tuned = TunedThresholdClassifierCV(
    estimator=model, scoring="f1", thresholds=100,
    cv=5, store_cv_results=True, random_state=42, n_jobs=-1
)
tuned.fit(X_tr, y_tr)

print(f"العتبة المثلى التي اختارها: {tuned.best_threshold_:.4f}")
print(f"أفضل F1 داخل التحقق المتقاطع: {tuned.best_score_:.4f}")
print(f"مفاتيح cv_results_: {list(tuned.cv_results_.keys())}")
print(f"أبعاد cv_results_['scores']: {tuned.cv_results_['scores'].shape}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer, confusion_matrix

def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 100 * fn + 10 * fp

# greater_is_better=False: سيتعظّم sklearn الناتج السالب أي يخفض التكلفة
cost_scorer = make_scorer(total_cost, greater_is_better=False, response_method="predict")

tuned_cost = TunedThresholdClassifierCV(
    estimator=model, scoring=cost_scorer, cv=5, random_state=42, n_jobs=-1
)
tuned_cost.fit(X_tr, y_tr)

y_default = model.predict(X_te)
y_optimized = tuned_cost.predict(X_te)
cost_default = total_cost(y_te, y_default)
cost_optimized = total_cost(y_te, y_optimized)

print(f"العتبة المثلى للتكلفة: {tuned_cost.best_threshold_:.4f}")
print(f"تكلفة العتبة الافتراضية 0.5: {cost_default}")
print(f"تكلفة العتبة المضبوطة:       {cost_optimized}")
print(f"التوفير على الاختبار: {cost_default - cost_optimized} وحدة")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import det_curve, DetCurveDisplay

hgb = HistGradientBoostingClassifier(max_iter=120, random_state=42).fit(X_tr, y_tr)

fpr_l, fnr_l, _ = det_curve(y_te, prob)                    # اللوجيستي
fpr_h, fnr_h, _ = det_curve(y_te, hgb.predict_proba(X_te)[:, 1])

order_l, order_h = np.argsort(fpr_l), np.argsort(fpr_h)
for target in [0.01, 0.05]:
    fnr_l_at = float(np.interp(target, fpr_l[order_l], fnr_l[order_l]))
    fnr_h_at = float(np.interp(target, fpr_h[order_h], fnr_h[order_h]))
    print(f"عند FPR = {target:.0%}: FNR اللوجيستي = {fnr_l_at:.4f} | FNR التدرّجي = {fnr_h_at:.4f}")

# للرسم الجاهز: DetCurveDisplay.from_estimator(hgb, X_te, y_te)

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
prob_rf = rf.predict_proba(X_te)[:, 1]

prob_true, prob_pred = calibration_curve(y_te, prob_rf, n_bins=8)
print("صندوق | متوسط معلَن | نسبة حقيقية")
for pred, true in zip(prob_pred, prob_true):
    print(f"      | {pred:13.3f} | {true:13.3f}")

mid = ((prob_rf > 0.1) & (prob_rf < 0.9)).mean()
print(f"نسبة الدرجات في المدى المتوسط 0.1-0.9: {mid:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV

def calibration_error(y, prob, n_bins=8):
    pt, pp = calibration_curve(y, prob, n_bins=n_bins)
    return float(np.abs(pt - pp).mean())

# عينة تدريب صغيرة: مختبر القاعدة العملية
X_small, _, y_small, _ = train_test_split(
    X_tr, y_tr, train_size=400, random_state=42, stratify=y_tr
)

cal_sig = CalibratedClassifierCV(model, method="sigmoid", cv=5).fit(X_small, y_small)
cal_iso = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_small, y_small)

err_sig = calibration_error(y_te, cal_sig.predict_proba(X_te)[:, 1])
err_iso = calibration_error(y_te, cal_iso.predict_proba(X_te)[:, 1])
print(f"خطأ المعايرة على بيانات صغيرة: sigmoid = {err_sig:.4f} | isotonic = {err_iso:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.frozen import FrozenEstimator

calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_tr, y_tr)
full_line = TunedThresholdClassifierCV(
    estimator=FrozenEstimator(calibrated), scoring="f1",
    cv=5, random_state=42, n_jobs=-1
)
full_line.fit(X_tr, y_tr)

y_line = full_line.predict(X_te)
print(f"عتبة الخط الكامل (فوق درجات معايرة): {full_line.best_threshold_:.4f}")
print(f"دقة الخط الكامل: {precision_score(y_te, y_line, zero_division=0):.4f}")
print(f"استرجاع الخط الكامل: {recall_score(y_te, y_line):.4f}")
print(f"تكلفة الخط الكامل: {total_cost(y_te, y_line)} | تكلفة العتبة 0.5 الخام: {total_cost(y_te, y_default)}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

# نستخرج العتبة من درجات خارج عينة على التدريب — لا من الاختبار (احترس من التسرب)
val_scores = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_tr, val_scores)
mask = rec[:-1] >= 0.9                 # rec تنازلي و thr أقصر بواحد
threshold_90 = float(thr[mask].min())  # أدنى عتبة تحقق الشرط = أعلى دقة

ftc90 = FixedThresholdClassifier(estimator=frozen_model, threshold=threshold_90)
y_hat90 = ftc90.predict(X_te)
print(f"العتبة المستخرجة لاسترجاع >= 0.9: {threshold_90:.4f}")
print(f"الاسترجاع الفعلي على الاختبار: {recall_score(y_te, y_hat90):.4f}")
print(f"الدقة عند هذه العتبة: {precision_score(y_te, y_hat90, zero_division=0):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] content/part09/ch77.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer

def spam_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 50 * fp + 5 * fn

spam_scorer = make_scorer(spam_cost, greater_is_better=False, response_method="predict")
tuned_spam = TunedThresholdClassifierCV(
    estimator=model, scoring=spam_scorer, cv=5, random_state=42, n_jobs=-1
)
tuned_spam.fit(X_tr, y_tr)
y_spam = tuned_spam.predict(X_te)

print(f"عتبة مزود البريد المثلى: {tuned_spam.best_threshold_:.4f}")
print(f"عتبة الطبيب (من القسم ٩.٤): {tuned_cost.best_threshold_:.4f}")
print(f"تكلفة مزود البريد عند عتبته: {spam_cost(y_te, y_spam)}")
print(f"لو استخدم عتبة الطبيب: {spam_cost(y_te, y_optimized)}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

X, y = make_classification(
    n_samples=4000, n_features=10, n_informative=4,
    weights=[0.95, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
prob = model.predict_proba(X_te)[:, 1]

print("العتبة | الاسترجاع | الدقة | الدقة الخام")
for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    y_hat = (prob >= t).astype(int)
    rec = recall_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    acc = accuracy_score(y_te, y_hat)
    print(f"{t:6.1f} | {rec:9.4f} | {prec:5.4f} | {acc:11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_score, recall_score

frozen_model = FrozenEstimator(model)   # نجمّد النموذج: لا إعادة تدريب مع كل عتبة

print("العتبة | الاسترجاع | الدقة | الدقة الخام")
for t in [0.2, 0.3, 0.5, 0.7]:
    ftc = FixedThresholdClassifier(estimator=frozen_model, threshold=t)
    y_hat = ftc.predict(X_te)          # قطع عند العتبة المطلوبة مباشرة
    rec = recall_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    acc = accuracy_score(y_te, y_hat)
    print(f"{t:6.1f} | {rec:9.4f} | {prec:5.4f} | {acc:11.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import TunedThresholdClassifierCV

tuned = TunedThresholdClassifierCV(
    estimator=model, scoring="f1", thresholds=100,
    cv=5, store_cv_results=True, random_state=42, n_jobs=-1
)
tuned.fit(X_tr, y_tr)

print(f"العتبة المثلى التي اختارها: {tuned.best_threshold_:.4f}")
print(f"أفضل F1 داخل التحقق المتقاطع: {tuned.best_score_:.4f}")
print(f"مفاتيح cv_results_: {list(tuned.cv_results_.keys())}")
print(f"أبعاد cv_results_['scores']: {tuned.cv_results_['scores'].shape}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer, confusion_matrix

def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 100 * fn + 10 * fp

# greater_is_better=False: سيتعظّم sklearn الناتج السالب أي يخفض التكلفة
cost_scorer = make_scorer(total_cost, greater_is_better=False, response_method="predict")

tuned_cost = TunedThresholdClassifierCV(
    estimator=model, scoring=cost_scorer, cv=5, random_state=42, n_jobs=-1
)
tuned_cost.fit(X_tr, y_tr)

y_default = model.predict(X_te)
y_optimized = tuned_cost.predict(X_te)
cost_default = total_cost(y_te, y_default)
cost_optimized = total_cost(y_te, y_optimized)

print(f"العتبة المثلى للتكلفة: {tuned_cost.best_threshold_:.4f}")
print(f"تكلفة العتبة الافتراضية 0.5: {cost_default}")
print(f"تكلفة العتبة المضبوطة:       {cost_optimized}")
print(f"التوفير على الاختبار: {cost_default - cost_optimized} وحدة")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import det_curve, DetCurveDisplay

hgb = HistGradientBoostingClassifier(max_iter=120, random_state=42).fit(X_tr, y_tr)

fpr_l, fnr_l, _ = det_curve(y_te, prob)                    # اللوجيستي
fpr_h, fnr_h, _ = det_curve(y_te, hgb.predict_proba(X_te)[:, 1])

order_l, order_h = np.argsort(fpr_l), np.argsort(fpr_h)
for target in [0.01, 0.05]:
    fnr_l_at = float(np.interp(target, fpr_l[order_l], fnr_l[order_l]))
    fnr_h_at = float(np.interp(target, fpr_h[order_h], fnr_h[order_h]))
    print(f"عند FPR = {target:.0%}: FNR اللوجيستي = {fnr_l_at:.4f} | FNR التدرّجي = {fnr_h_at:.4f}")

# للرسم الجاهز: DetCurveDisplay.from_estimator(hgb, X_te, y_te)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
prob_rf = rf.predict_proba(X_te)[:, 1]

prob_true, prob_pred = calibration_curve(y_te, prob_rf, n_bins=8)
print("صندوق | متوسط معلَن | نسبة حقيقية")
for pred, true in zip(prob_pred, prob_true):
    print(f"      | {pred:13.3f} | {true:13.3f}")

mid = ((prob_rf > 0.1) & (prob_rf < 0.9)).mean()
print(f"نسبة الدرجات في المدى المتوسط 0.1-0.9: {mid:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV

def calibration_error(y, prob, n_bins=8):
    pt, pp = calibration_curve(y, prob, n_bins=n_bins)
    return float(np.abs(pt - pp).mean())

# عينة تدريب صغيرة: مختبر القاعدة العملية
X_small, _, y_small, _ = train_test_split(
    X_tr, y_tr, train_size=400, random_state=42, stratify=y_tr
)

cal_sig = CalibratedClassifierCV(model, method="sigmoid", cv=5).fit(X_small, y_small)
cal_iso = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_small, y_small)

err_sig = calibration_error(y_te, cal_sig.predict_proba(X_te)[:, 1])
err_iso = calibration_error(y_te, cal_iso.predict_proba(X_te)[:, 1])
print(f"خطأ المعايرة على بيانات صغيرة: sigmoid = {err_sig:.4f} | isotonic = {err_iso:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.frozen import FrozenEstimator

calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_tr, y_tr)
full_line = TunedThresholdClassifierCV(
    estimator=FrozenEstimator(calibrated), scoring="f1",
    cv=5, random_state=42, n_jobs=-1
)
full_line.fit(X_tr, y_tr)

y_line = full_line.predict(X_te)
print(f"عتبة الخط الكامل (فوق درجات معايرة): {full_line.best_threshold_:.4f}")
print(f"دقة الخط الكامل: {precision_score(y_te, y_line, zero_division=0):.4f}")
print(f"استرجاع الخط الكامل: {recall_score(y_te, y_line):.4f}")
print(f"تكلفة الخط الكامل: {total_cost(y_te, y_line)} | تكلفة العتبة 0.5 الخام: {total_cost(y_te, y_default)}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

# نستخرج العتبة من درجات خارج عينة على التدريب — لا من الاختبار (احترس من التسرب)
val_scores = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_tr, val_scores)
mask = rec[:-1] >= 0.9                 # rec تنازلي و thr أقصر بواحد
threshold_90 = float(thr[mask].min())  # أدنى عتبة تحقق الشرط = أعلى دقة

ftc90 = FixedThresholdClassifier(estimator=frozen_model, threshold=threshold_90)
y_hat90 = ftc90.predict(X_te)
print(f"العتبة المستخرجة لاسترجاع >= 0.9: {threshold_90:.4f}")
print(f"الاسترجاع الفعلي على الاختبار: {recall_score(y_te, y_hat90):.4f}")
print(f"الدقة عند هذه العتبة: {precision_score(y_te, y_hat90, zero_division=0):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-77-ch77.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer

def spam_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 50 * fp + 5 * fn

spam_scorer = make_scorer(spam_cost, greater_is_better=False, response_method="predict")
tuned_spam = TunedThresholdClassifierCV(
    estimator=model, scoring=spam_scorer, cv=5, random_state=42, n_jobs=-1
)
tuned_spam.fit(X_tr, y_tr)
y_spam = tuned_spam.predict(X_te)

print(f"عتبة مزود البريد المثلى: {tuned_spam.best_threshold_:.4f}")
print(f"عتبة الطبيب (من القسم ٩.٤): {tuned_cost.best_threshold_:.4f}")
print(f"تكلفة مزود البريد عند عتبته: {spam_cost(y_te, y_spam)}")
print(f"لو استخدم عتبة الطبيب: {spam_cost(y_te, y_optimized)}")
