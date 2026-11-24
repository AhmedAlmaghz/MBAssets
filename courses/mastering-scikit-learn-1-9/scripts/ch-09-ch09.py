#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أخلاقيات التطبيق وحدود الخوارزميات
Mastering scikit-learn 1.9 — Chapter 09 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-09-ch09 (lesson 9)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-09-ch09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — أخلاقيات التطبيق وحدود الخوارزميات
Mastering scikit-learn 1.9 — Chapter 09 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-09-ch09 (lesson 9)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch09.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n_samples = 1500

# السمة الحساسة A (مثلاً: 0 تمثل الفئة الأقلية، 1 تمثل الفئة الأغلبية)
group_A = rng.integers(0, 2, n_samples)

# ميزة مالية موضوعية: الدرجة الائتمانية (تتأثر جزئياً بظروف تاريخية غير متكافئة)
credit_score = rng.normal(600, 50, n_samples) + (group_A * 25)

# احتمالية سداد القرض الحقيقية تعتمد على الجدارة المالية
repay_prob = 1 / (1 + np.exp(-(credit_score - 600) / 45))
loan_approved = rng.binomial(1, repay_prob)

X = np.column_stack([credit_score, group_A])
X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, loan_approved, group_A, test_size=0.3, random_state=42
)

# تدريب مصنّف لوجيستي عادي
clf = LogisticRegression().fit(X_tr, y_tr)
predictions = clf.predict(X_te)

# قياس معدل القبول لكل فئة
rate_group_0 = predictions[g_te == 0].mean()
rate_group_1 = predictions[g_te == 1].mean()
dp_diff = abs(rate_group_0 - rate_group_1)
disparate_impact = rate_group_0 / rate_group_1

print(f"معدل قبول الفئة الأولى (A=0): {rate_group_0:.1%}")
print(f"معدل قبول الفئة الثانية (A=1): {rate_group_1:.1%}")
print(f"فارق التكافؤ الديموغرافي (DP Difference): {dp_diff:.3f}")
print(f"نسبة الأثر المتباين (Disparate Impact)   : {disparate_impact:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch09.md
# ────────────────────────────────────────────────────────────────────────
# محاكاة متغير وسيط (مثل الرمز البريدي) يرتبط بقوة بالسمة الحساسة
zip_code_score = credit_score * 0.8 + group_A * 40 + rng.normal(0, 10, n_samples)

# مصفوفة تدريب جديدة خالية تماماً من العمود الحساس group_A
X_no_sensitive = np.column_stack([credit_score, zip_code_score])
X_tr_ns, X_te_ns, _, _ = train_test_split(
    X_no_sensitive, loan_approved, test_size=0.3, random_state=42
)

# تدريب النموذج على البيانات «المعمية»
clf_blind = LogisticRegression().fit(X_tr_ns, y_tr)
preds_blind = clf_blind.predict(X_te_ns)

rate_blind_0 = preds_blind[g_te == 0].mean()
rate_blind_1 = preds_blind[g_te == 1].mean()

print("نتائج النموذج الأعمى بعد حذف السمة الحساسة تماماً:")
print(f"  - معدل قبول الفئة A=0 : {rate_blind_0:.1%}")
print(f"  - معدل قبول الفئة A=1 : {rate_blind_1:.1%}")
print(f"  - فارق التكافؤ الباقي   : {abs(rate_blind_0 - rate_blind_1):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch09.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import recall_score

# حساب الاسترجاع (TPR) لكل فئة على حدة بين المستحقين فعلياً للسداد
tpr_0 = recall_score(y_te[g_te == 0], predictions[g_te == 0])
tpr_1 = recall_score(y_te[g_te == 1], predictions[g_te == 1])
eo_diff = abs(tpr_0 - tpr_1)

print("تدقيق تكافؤ الفرص (True Positive Rate per Group):")
print(f"  - نسبة قبول المؤهلين فعلياً من الفئة A=0: {tpr_0:.1%}")
print(f"  - نسبة قبول المؤهلين فعلياً من الفئة A=1: {tpr_1:.1%}")
print(f"  - فارق تكافؤ الفرص (Equal Opportunity Diff): {eo_diff:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch09.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import TunedThresholdClassifierCV

# تغليف المصنّف لضبط عتبة القرار تلقائياً باستخدام الدقة المتوازنة
tuned_model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(),
    scoring="balanced_accuracy",
    cv=5
)

tuned_model.fit(X_tr, y_tr)
optimal_threshold = tuned_model.best_threshold_
tuned_preds = tuned_model.predict(X_te)

print("نتائج ضبط العتبة التلقائي في scikit-learn 1.9:")
print(f"  - عتبة القرار المحسوبة كأمثل خيار: {optimal_threshold:.4f} (بدلاً من 0.5000)")
print(f"  - معدل القبول الجديد للفئة A=0   : {tuned_preds[g_te == 0].mean():.1%}")
print(f"  - معدل القبول الجديد للفئة A=1   : {tuned_preds[g_te == 1].mean():.1%}")
print(f"  - الفارق الجديد في القبول       : {abs(tuned_preds[g_te == 0].mean() - tuned_preds[g_te == 1].mean()):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch09.md
# ────────────────────────────────────────────────────────────────────────
def audit_fairness(model, X_eval, y_eval, sensitive_col_idx):
    """دالة قياسية تفحص التكافؤ وتكافؤ الفرص عبر الفئات."""
    preds = model.predict(X_eval)
    groups = np.unique(X_eval[:, sensitive_col_idx])

    report = {}
    for g in groups:
        mask = (X_eval[:, sensitive_col_idx] == g)
        pos_rate = preds[mask].mean()
        recall = recall_score(y_eval[mask], preds[mask]) if y_eval[mask].sum() > 0 else 0.0
        report[f"Group_{int(g)}"] = {"Selection_Rate": round(pos_rate, 3), "TPR_Recall": round(recall, 3)}
    return report

fairness_report = audit_fairness(tuned_model, X_te, y_te, sensitive_col_idx=1)
print("تقرير التدقيق الأخلاقي السريع للنموذج:")
for grp, metrics in fairness_report.items():
    print(f"  {grp}: معدل الاختيار = {metrics['Selection_Rate']:.1%}, الاسترجاع = {metrics['TPR_Recall']:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-09-ch09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n_samples = 1500

# السمة الحساسة A (مثلاً: 0 تمثل الفئة الأقلية، 1 تمثل الفئة الأغلبية)
group_A = rng.integers(0, 2, n_samples)

# ميزة مالية موضوعية: الدرجة الائتمانية (تتأثر جزئياً بظروف تاريخية غير متكافئة)
credit_score = rng.normal(600, 50, n_samples) + (group_A * 25)

# احتمالية سداد القرض الحقيقية تعتمد على الجدارة المالية
repay_prob = 1 / (1 + np.exp(-(credit_score - 600) / 45))
loan_approved = rng.binomial(1, repay_prob)

X = np.column_stack([credit_score, group_A])
X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, loan_approved, group_A, test_size=0.3, random_state=42
)

# تدريب مصنّف لوجيستي عادي
clf = LogisticRegression().fit(X_tr, y_tr)
predictions = clf.predict(X_te)

# قياس معدل القبول لكل فئة
rate_group_0 = predictions[g_te == 0].mean()
rate_group_1 = predictions[g_te == 1].mean()
dp_diff = abs(rate_group_0 - rate_group_1)
disparate_impact = rate_group_0 / rate_group_1

print(f"معدل قبول الفئة الأولى (A=0): {rate_group_0:.1%}")
print(f"معدل قبول الفئة الثانية (A=1): {rate_group_1:.1%}")
print(f"فارق التكافؤ الديموغرافي (DP Difference): {dp_diff:.3f}")
print(f"نسبة الأثر المتباين (Disparate Impact)   : {disparate_impact:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-09-ch09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# محاكاة متغير وسيط (مثل الرمز البريدي) يرتبط بقوة بالسمة الحساسة
zip_code_score = credit_score * 0.8 + group_A * 40 + rng.normal(0, 10, n_samples)

# مصفوفة تدريب جديدة خالية تماماً من العمود الحساس group_A
X_no_sensitive = np.column_stack([credit_score, zip_code_score])
X_tr_ns, X_te_ns, _, _ = train_test_split(
    X_no_sensitive, loan_approved, test_size=0.3, random_state=42
)

# تدريب النموذج على البيانات «المعمية»
clf_blind = LogisticRegression().fit(X_tr_ns, y_tr)
preds_blind = clf_blind.predict(X_te_ns)

rate_blind_0 = preds_blind[g_te == 0].mean()
rate_blind_1 = preds_blind[g_te == 1].mean()

print("نتائج النموذج الأعمى بعد حذف السمة الحساسة تماماً:")
print(f"  - معدل قبول الفئة A=0 : {rate_blind_0:.1%}")
print(f"  - معدل قبول الفئة A=1 : {rate_blind_1:.1%}")
print(f"  - فارق التكافؤ الباقي   : {abs(rate_blind_0 - rate_blind_1):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-09-ch09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import recall_score

# حساب الاسترجاع (TPR) لكل فئة على حدة بين المستحقين فعلياً للسداد
tpr_0 = recall_score(y_te[g_te == 0], predictions[g_te == 0])
tpr_1 = recall_score(y_te[g_te == 1], predictions[g_te == 1])
eo_diff = abs(tpr_0 - tpr_1)

print("تدقيق تكافؤ الفرص (True Positive Rate per Group):")
print(f"  - نسبة قبول المؤهلين فعلياً من الفئة A=0: {tpr_0:.1%}")
print(f"  - نسبة قبول المؤهلين فعلياً من الفئة A=1: {tpr_1:.1%}")
print(f"  - فارق تكافؤ الفرص (Equal Opportunity Diff): {eo_diff:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-09-ch09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import TunedThresholdClassifierCV

# تغليف المصنّف لضبط عتبة القرار تلقائياً باستخدام الدقة المتوازنة
tuned_model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(),
    scoring="balanced_accuracy",
    cv=5
)

tuned_model.fit(X_tr, y_tr)
optimal_threshold = tuned_model.best_threshold_
tuned_preds = tuned_model.predict(X_te)

print("نتائج ضبط العتبة التلقائي في scikit-learn 1.9:")
print(f"  - عتبة القرار المحسوبة كأمثل خيار: {optimal_threshold:.4f} (بدلاً من 0.5000)")
print(f"  - معدل القبول الجديد للفئة A=0   : {tuned_preds[g_te == 0].mean():.1%}")
print(f"  - معدل القبول الجديد للفئة A=1   : {tuned_preds[g_te == 1].mean():.1%}")
print(f"  - الفارق الجديد في القبول       : {abs(tuned_preds[g_te == 0].mean() - tuned_preds[g_te == 1].mean()):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-09-ch09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def audit_fairness(model, X_eval, y_eval, sensitive_col_idx):
    """دالة قياسية تفحص التكافؤ وتكافؤ الفرص عبر الفئات."""
    preds = model.predict(X_eval)
    groups = np.unique(X_eval[:, sensitive_col_idx])

    report = {}
    for g in groups:
        mask = (X_eval[:, sensitive_col_idx] == g)
        pos_rate = preds[mask].mean()
        recall = recall_score(y_eval[mask], preds[mask]) if y_eval[mask].sum() > 0 else 0.0
        report[f"Group_{int(g)}"] = {"Selection_Rate": round(pos_rate, 3), "TPR_Recall": round(recall, 3)}
    return report

fairness_report = audit_fairness(tuned_model, X_te, y_te, sensitive_col_idx=1)
print("تقرير التدقيق الأخلاقي السريع للنموذج:")
for grp, metrics in fairness_report.items():
    print(f"  {grp}: معدل الاختيار = {metrics['Selection_Rate']:.1%}, الاسترجاع = {metrics['TPR_Recall']:.1%}")
