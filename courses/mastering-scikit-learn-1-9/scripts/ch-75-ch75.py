#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس التصنيف: ما بعد الدقة
Mastering scikit-learn 1.9 — Chapter 75 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-75-ch75 (lesson 75)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-75-ch75.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس التصنيف: ما بعد الدقة
Mastering scikit-learn 1.9 — Chapter 75 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-75-ch75 (lesson 75)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

n_patients = 10000
n_sick = 100  # نسبة الانتشار 1%

y_true = np.zeros(n_patients, dtype=int)
y_true[:n_sick] = 1            # أول 100 مريض مصاب

y_naive = np.zeros(n_patients, dtype=int)  # «لا مرض» للجميع بلا استثناء

accuracy = (y_true == y_naive).mean()                    # الدقة
recall = (y_naive[y_true == 1] == 1).mean()              # ماذا اكتشف من المصابين؟

print(f"الدقة accuracy:   {accuracy:.2%} — رقم يفتح الأبواب في العروض!")
print(f"الاسترجاع recall: {recall:.2%} — لم يكتشف مريضاً واحداً!")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X, y = load_breast_cancer(return_X_y=True)   # 0: خبيث ، 1: حميد
y_sick = (y == 0).astype(int)                # نعيد التعريف: 1 = خبيث (إيجابي)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

cm = confusion_matrix(y_te, y_pred)   # الصفوف: الحقيقة ، الأعمدة: التوقع
tn, fp, fn, tp = cm.ravel()
print(f"TP = {tp:3d} (خبيث اكتُشف)     FN = {fn} (خبيث تفوّت!)")
print(f"FP = {fp:3d} (إنذار كاذب)     TN = {tn} (حميد طُابق)")
print(f"الدقة الخام: {(tp + tn) / cm.sum():.4f}")

# والمعامل labels يعيد الترتيب كما تريد: الخبيث أولاً
cm_flipped = confusion_matrix(y_te, y_pred, labels=[1, 0])
print("بالترتيب labels=[1, 0] تصبح الخانة [0, 0] هي TP:", cm_flipped[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score,
)

# عيادة في يوم واحد: 200 مريض منهم 50 مصاباً
TP, FN = 40, 10
TN, FP = 135, 15

precision_manual = TP / (TP + FP)          # 40 / 55
recall_manual    = TP / (TP + FN)          # 40 / 50
f1_manual        = 2 * TP / (2 * TP + FP + FN)
spec_manual      = TN / (TN + FP)          # الخصوصية
ba_manual        = (recall_manual + spec_manual) / 2

# إعادة بناء اليوم نفسه كمتجهات والتحقق من sklearn
y_true = np.array([1] * 50 + [0] * 150)
y_pred = np.array([1] * 40 + [0] * 10 + [1] * 15 + [0] * 135)

print(f"الدقة precision:  يدوياً {precision_manual:.4f} | sklearn {precision_score(y_true, y_pred):.4f}")
print(f"الاسترجاع recall: يدوياً {recall_manual:.4f} | sklearn {recall_score(y_true, y_pred):.4f}")
print(f"مقياس F1:         يدوياً {f1_manual:.4f} | sklearn {f1_score(y_true, y_pred):.4f}")
print(f"الخصوصية:         يدوياً {spec_manual:.4f} | الدقة الخام {accuracy_score(y_true, y_pred):.4f}")
print(f"الدقة المتوازنة:  يدوياً {ba_manual:.4f} | sklearn {balanced_accuracy_score(y_true, y_pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score

y_true = np.array([1] * 100 + [0] * 9900)

# المصنف أ: «لا مرض دائماً» — دقة 99% بلا أي كشف
y_pred_a = np.zeros(10000, dtype=int)

# المصنف ب: يكتشف 50 من 100 مصاب مقابل 50 إنذاراً كاذباً
y_pred_b = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 9850)

for name, y_pred in [("أ (لا مرض دائماً)", y_pred_a), ("ب (متحفظ حقيقي)", y_pred_b)]:
    acc = (y_true == y_pred).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    ba  = balanced_accuracy_score(y_true, y_pred)
    f1b = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name} | الدقة: {acc:.2%} | MCC: {mcc:.4f} | متوازنة: {ba:.4f} | F1: {f1b:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

X, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5, n_classes=3,
    weights=[0.80, 0.15, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

for avg in ["macro", "micro", "weighted"]:
    print(f"F1 (average={avg:9s}): {f1_score(y_te, y_pred, average=avg):.4f}")
print(f"مطابقة micro مع الدقة الخام: {f1_score(y_te, y_pred, average='micro') == accuracy_score(y_te, y_pred)}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, precision_recall_fscore_support

print(classification_report(
    y_te, y_pred,
    target_names=["سليم", "شبه مصاب", "مصاب"],
    digits=4
))

# الأداة الخام خلف التقرير: درجات لكل فئة بلا تجميع
prec, rec, f1s, support = precision_recall_fscore_support(y_te, y_pred)
print("مصفوفة الدقة لكل فئة:", prec.round(4))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_score

y_true = np.array(["مريض", "سليم", "مريض", "سليم", "مريض", "سليم"])
y_pred = np.array(["مريض", "مريض", "مريض", "سليم", "سليم", "سليم"])

try:
    precision_score(y_true, y_pred)          # يبحث عن pos_label=1 الافتراضي
except ValueError as err:
    print(f"ValueError: {err}")

# الحل: حدد الفئة الإيجابية بنصها
p = precision_score(y_true, y_pred, pos_label="مريض")
print(f"بعد التحديد الصحيح pos_label='مريض': الدقة = {p:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
)

# 80 صحيحاً من 120 المعلَّمة ، و 20 مزعجاً تفوّتا ، و 860 سليماً طُابق
TP, FP, FN, TN = 80, 40, 20, 860

y_true = np.array([1] * 100 + [0] * 900)
y_pred = np.array([1] * 80 + [0] * 20 + [1] * 40 + [0] * 860)

mcc_manual = (TP * TN - FP * FN) / np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
)

print(f"الدقة precision: {TP / (TP + FP):.4f} | {precision_score(y_true, y_pred):.4f}")
print(f"الاسترجاع recall: {TP / (TP + FN):.4f} | {recall_score(y_true, y_pred):.4f}")
print(f"مقياس F1:        {2 * TP / (2 * TP + FP + FN):.4f} | {f1_score(y_true, y_pred):.4f}")
print(f"الخصوصية:        {TN / (TN + FP):.4f}")
print(f"MCC:             {mcc_manual:.4f} | {matthews_corrcoef(y_true, y_pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] content/part09/ch75.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
y_sick = (y == 0).astype(int)          # الخبيث إيجابي كما في القسم ٩.٢
X_tr, X_te, ys_tr, ys_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)
bc_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
bc_model.fit(X_tr, ys_tr)
probabilities = bc_model.predict_proba(X_te)[:, 1]

print("العتبة | الدقة precision | الاسترجاع recall")
for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    y_hat = (probabilities >= t).astype(int)
    p = precision_score(ys_te, y_hat, zero_division=0)
    r = recall_score(ys_te, y_hat)
    print(f"{t:6.1f} | {p:14.4f} | {r:14.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

n_patients = 10000
n_sick = 100  # نسبة الانتشار 1%

y_true = np.zeros(n_patients, dtype=int)
y_true[:n_sick] = 1            # أول 100 مريض مصاب

y_naive = np.zeros(n_patients, dtype=int)  # «لا مرض» للجميع بلا استثناء

accuracy = (y_true == y_naive).mean()                    # الدقة
recall = (y_naive[y_true == 1] == 1).mean()              # ماذا اكتشف من المصابين؟

print(f"الدقة accuracy:   {accuracy:.2%} — رقم يفتح الأبواب في العروض!")
print(f"الاسترجاع recall: {recall:.2%} — لم يكتشف مريضاً واحداً!")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X, y = load_breast_cancer(return_X_y=True)   # 0: خبيث ، 1: حميد
y_sick = (y == 0).astype(int)                # نعيد التعريف: 1 = خبيث (إيجابي)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

cm = confusion_matrix(y_te, y_pred)   # الصفوف: الحقيقة ، الأعمدة: التوقع
tn, fp, fn, tp = cm.ravel()
print(f"TP = {tp:3d} (خبيث اكتُشف)     FN = {fn} (خبيث تفوّت!)")
print(f"FP = {fp:3d} (إنذار كاذب)     TN = {tn} (حميد طُابق)")
print(f"الدقة الخام: {(tp + tn) / cm.sum():.4f}")

# والمعامل labels يعيد الترتيب كما تريد: الخبيث أولاً
cm_flipped = confusion_matrix(y_te, y_pred, labels=[1, 0])
print("بالترتيب labels=[1, 0] تصبح الخانة [0, 0] هي TP:", cm_flipped[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score,
)

# عيادة في يوم واحد: 200 مريض منهم 50 مصاباً
TP, FN = 40, 10
TN, FP = 135, 15

precision_manual = TP / (TP + FP)          # 40 / 55
recall_manual    = TP / (TP + FN)          # 40 / 50
f1_manual        = 2 * TP / (2 * TP + FP + FN)
spec_manual      = TN / (TN + FP)          # الخصوصية
ba_manual        = (recall_manual + spec_manual) / 2

# إعادة بناء اليوم نفسه كمتجهات والتحقق من sklearn
y_true = np.array([1] * 50 + [0] * 150)
y_pred = np.array([1] * 40 + [0] * 10 + [1] * 15 + [0] * 135)

print(f"الدقة precision:  يدوياً {precision_manual:.4f} | sklearn {precision_score(y_true, y_pred):.4f}")
print(f"الاسترجاع recall: يدوياً {recall_manual:.4f} | sklearn {recall_score(y_true, y_pred):.4f}")
print(f"مقياس F1:         يدوياً {f1_manual:.4f} | sklearn {f1_score(y_true, y_pred):.4f}")
print(f"الخصوصية:         يدوياً {spec_manual:.4f} | الدقة الخام {accuracy_score(y_true, y_pred):.4f}")
print(f"الدقة المتوازنة:  يدوياً {ba_manual:.4f} | sklearn {balanced_accuracy_score(y_true, y_pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score

y_true = np.array([1] * 100 + [0] * 9900)

# المصنف أ: «لا مرض دائماً» — دقة 99% بلا أي كشف
y_pred_a = np.zeros(10000, dtype=int)

# المصنف ب: يكتشف 50 من 100 مصاب مقابل 50 إنذاراً كاذباً
y_pred_b = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 9850)

for name, y_pred in [("أ (لا مرض دائماً)", y_pred_a), ("ب (متحفظ حقيقي)", y_pred_b)]:
    acc = (y_true == y_pred).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    ba  = balanced_accuracy_score(y_true, y_pred)
    f1b = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name} | الدقة: {acc:.2%} | MCC: {mcc:.4f} | متوازنة: {ba:.4f} | F1: {f1b:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

X, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5, n_classes=3,
    weights=[0.80, 0.15, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

for avg in ["macro", "micro", "weighted"]:
    print(f"F1 (average={avg:9s}): {f1_score(y_te, y_pred, average=avg):.4f}")
print(f"مطابقة micro مع الدقة الخام: {f1_score(y_te, y_pred, average='micro') == accuracy_score(y_te, y_pred)}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, precision_recall_fscore_support

print(classification_report(
    y_te, y_pred,
    target_names=["سليم", "شبه مصاب", "مصاب"],
    digits=4
))

# الأداة الخام خلف التقرير: درجات لكل فئة بلا تجميع
prec, rec, f1s, support = precision_recall_fscore_support(y_te, y_pred)
print("مصفوفة الدقة لكل فئة:", prec.round(4))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_score

y_true = np.array(["مريض", "سليم", "مريض", "سليم", "مريض", "سليم"])
y_pred = np.array(["مريض", "مريض", "مريض", "سليم", "سليم", "سليم"])

try:
    precision_score(y_true, y_pred)          # يبحث عن pos_label=1 الافتراضي
except ValueError as err:
    print(f"ValueError: {err}")

# الحل: حدد الفئة الإيجابية بنصها
p = precision_score(y_true, y_pred, pos_label="مريض")
print(f"بعد التحديد الصحيح pos_label='مريض': الدقة = {p:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
)

# 80 صحيحاً من 120 المعلَّمة ، و 20 مزعجاً تفوّتا ، و 860 سليماً طُابق
TP, FP, FN, TN = 80, 40, 20, 860

y_true = np.array([1] * 100 + [0] * 900)
y_pred = np.array([1] * 80 + [0] * 20 + [1] * 40 + [0] * 860)

mcc_manual = (TP * TN - FP * FN) / np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
)

print(f"الدقة precision: {TP / (TP + FP):.4f} | {precision_score(y_true, y_pred):.4f}")
print(f"الاسترجاع recall: {TP / (TP + FN):.4f} | {recall_score(y_true, y_pred):.4f}")
print(f"مقياس F1:        {2 * TP / (2 * TP + FP + FN):.4f} | {f1_score(y_true, y_pred):.4f}")
print(f"الخصوصية:        {TN / (TN + FP):.4f}")
print(f"MCC:             {mcc_manual:.4f} | {matthews_corrcoef(y_true, y_pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-75-ch75.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
y_sick = (y == 0).astype(int)          # الخبيث إيجابي كما في القسم ٩.٢
X_tr, X_te, ys_tr, ys_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)
bc_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
bc_model.fit(X_tr, ys_tr)
probabilities = bc_model.predict_proba(X_te)[:, 1]

print("العتبة | الدقة precision | الاسترجاع recall")
for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    y_hat = (probabilities >= t).astype(int)
    p = precision_score(ys_te, y_hat, zero_division=0)
    r = recall_score(ys_te, y_hat)
    print(f"{t:6.1f} | {p:14.4f} | {r:14.4f}")
