#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منحنيات القرار و ROC و PR
Mastering scikit-learn 1.9 — ROC

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-76-ch76 (lesson 76)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-76-ch76.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — منحنيات القرار و ROC و PR
Mastering scikit-learn 1.9 — ROC

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-76-ch76 (lesson 76)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
y_sick = (y == 0).astype(int)          # 1 = خبيث (إيجابي) كما اتفقنا في الفصل ٧٥
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)

prob = model.predict_proba(X_te)[:, 1]      # درجات الاحتمال الخام
d_05 = (prob >= 0.5)                        # قرار العتبة الافتراضية
d_03 = (prob >= 0.3)                        # قرار عتبة أخفض

print(f"إعلانات خباثة عند العتبة 0.5: {d_05.sum()} من {len(prob)}")
print(f"إعلانات خباثة عند العتبة 0.3: {d_03.sum()} من {len(prob)}")
print(f"قرارات انقلبت بينهما: {(d_05 != d_03).sum()}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_curve

y_true = np.array([1, 1, 0, 1, 0, 0])
scores = np.array([0.9, 0.6, 0.55, 0.4, 0.3, 0.2])

# نقطة يدوية عند العتبة 0.5: من تجاوزها؟
y_hat = (scores >= 0.5).astype(int)
tp = int(((y_true == 1) & (y_hat == 1)).sum())
fp = int(((y_true == 0) & (y_hat == 1)).sum())
fn = int(((y_true == 1) & (y_hat == 0)).sum())
tn = int(((y_true == 0) & (y_hat == 0)).sum())
print(f"عند العتبة 0.5: TPR = {tp}/{tp + fn} = {tp / (tp + fn):.4f} ، FPR = {fp}/{fp + tn} = {fp / (fp + tn):.4f}")

# المنحنى الكامل كما يحسبه sklearn
fpr, tpr, thresholds = roc_curve(y_true, scores)
print("العتبات التي فحصها:", thresholds)
print("FPR:", fpr.round(4))
print("TPR:", tpr.round(4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([1, 1, 0, 1, 0, 0])
scores = np.array([0.9, 0.6, 0.55, 0.4, 0.3, 0.2])

pos = scores[y_true == 1]
neg = scores[y_true == 0]
wins = sum(1 for p in pos for q in neg if p > q)
auc_manual = wins / (len(pos) * len(neg))

print(f"الأزواج الرابحة: {wins} من {len(pos) * len(neg)} أزواج")
print(f"AUC بالعد اليدوي: {auc_manual:.4f}")
print(f"AUC من roc_auc_score: {roc_auc_score(y_true, scores):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)

print(f"AUC متعدد الفئات (ovr, macro):  {roc_auc_score(y_te, proba, multi_class='ovr', average='macro'):.4f}")
print(f"AUC متعدد الفئات (ovr, micro):  {roc_auc_score(y_te, proba, multi_class='ovr', average='micro'):.4f}")
print(f"AUC متعدد الفئات (ovo, macro):  {roc_auc_score(y_te, proba, multi_class='ovo', average='macro'):.4f}")

# الخطأ الشائع: تمرير درجات decision_function بدل الاحتمالات
try:
    roc_auc_score(y_te, model.decision_function(X_te), multi_class='ovr')
except ValueError as err:
    print("ValueError:", str(err)[:100])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)

X, y = make_classification(
    n_samples=20000, n_features=12, n_informative=6, n_redundant=2,
    weights=[0.99, 0.01], class_sep=1.5, flip_y=0.0, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"نسبة الإيجابيات في الاختبار: {y_te.mean():.2%}")

log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
s_log = log.predict_proba(X_te)[:, 1]
hgb = HistGradientBoostingClassifier(max_iter=120, random_state=42).fit(X_tr, y_tr)
s_hgb = hgb.predict_proba(X_te)[:, 1]

print(f"اللوجيستي: AUC = {roc_auc_score(y_te, s_log):.4f} | AP = {average_precision_score(y_te, s_log):.4f}")
print(f"التدرّجي:  AUC = {roc_auc_score(y_te, s_hgb):.4f} | AP = {average_precision_score(y_te, s_hgb):.4f}")

# حقيقة التشغيل: الدقة الفعلية عند مستويات استرجاع نحتاجها
import numpy as np
from sklearn.metrics import precision_recall_curve

prec, rec, _ = precision_recall_curve(y_te, s_log)
order = np.argsort(rec)                  # الاسترجاع تنازلي: نرتّبه صعودياً للاستيفاء
for r in [0.5, 0.9]:
    p_at_r = float(np.interp(r, rec[order], prec[order]))
    print(f"اللوجيستي عند استرجاع {r:.0%}: الدقة الفعلية = {p_at_r:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
constant_scores = np.full(len(y_te), dummy.predict_proba(X_te)[0, 1])

print(f"معدل الانتشار: {y_te.mean():.4f}")
print(f"AUC للدرجات الثابتة: {roc_auc_score(y_te, constant_scores):.4f} (القطر)")
print(f"AP للدرجات الثابتة:  {average_precision_score(y_te, constant_scores):.4f} (خط أفقي)")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import det_curve

fpr_l, fnr_l, _ = det_curve(y_te, s_log)
fpr_h, fnr_h, _ = det_curve(y_te, s_hgb)

order_l = np.argsort(fpr_l)
order_h = np.argsort(fpr_h)
for target_fpr in [0.01, 0.05]:
    fnr_at_l = float(np.interp(target_fpr, fpr_l[order_l], fnr_l[order_l]))
    fnr_at_h = float(np.interp(target_fpr, fpr_h[order_h], fnr_h[order_h]))
    print(f"عند FPR = {target_fpr:.0%}: FNR اللوجيستي = {fnr_at_l:.4f} | FNR التدرّجي = {fnr_at_h:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
scores = np.array([0.85, 0.7, 0.6, 0.5, 0.45, 0.3, 0.25, 0.1])

pos = scores[y_true == 1]
neg = scores[y_true == 0]
wins = sum(1 for p in pos for q in neg if p > q)

print(f"الأزواج الرابحة: {wins} من {len(pos) * len(neg)}")
print(f"AUC يدوياً: {wins / (len(pos) * len(neg)):.4f} | roc_auc_score: {roc_auc_score(y_true, scores):.4f}")
fpr, tpr, thr = roc_curve(y_true, scores)
print(f"عدد نقاط المنحنى: {len(fpr)} | عدد العتبات: {len(thr)}")
print("FPR:", fpr.round(4))
print("TPR:", tpr.round(4))

# ────────────────────────────────────────────────────────────────────────
# [009] content/part09/ch76.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
random_scores = rng.random(2000)
y_random = (rng.random(2000) < 0.05).astype(int)   # انتشار 5%

print(f"الانتشار: {y_random.mean():.4f}")
print(f"AUC درجات عشوائية: {roc_auc_score(y_random, random_scores):.4f} (المفترض: 0.5)")
print(f"AP درجات عشوائية:  {average_precision_score(y_random, random_scores):.4f} (المفترض: الانتشار)")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
y_sick = (y == 0).astype(int)          # 1 = خبيث (إيجابي) كما اتفقنا في الفصل ٧٥
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_sick, test_size=0.25, random_state=42, stratify=y_sick
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)

prob = model.predict_proba(X_te)[:, 1]      # درجات الاحتمال الخام
d_05 = (prob >= 0.5)                        # قرار العتبة الافتراضية
d_03 = (prob >= 0.3)                        # قرار عتبة أخفض

print(f"إعلانات خباثة عند العتبة 0.5: {d_05.sum()} من {len(prob)}")
print(f"إعلانات خباثة عند العتبة 0.3: {d_03.sum()} من {len(prob)}")
print(f"قرارات انقلبت بينهما: {(d_05 != d_03).sum()}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_curve

y_true = np.array([1, 1, 0, 1, 0, 0])
scores = np.array([0.9, 0.6, 0.55, 0.4, 0.3, 0.2])

# نقطة يدوية عند العتبة 0.5: من تجاوزها؟
y_hat = (scores >= 0.5).astype(int)
tp = int(((y_true == 1) & (y_hat == 1)).sum())
fp = int(((y_true == 0) & (y_hat == 1)).sum())
fn = int(((y_true == 1) & (y_hat == 0)).sum())
tn = int(((y_true == 0) & (y_hat == 0)).sum())
print(f"عند العتبة 0.5: TPR = {tp}/{tp + fn} = {tp / (tp + fn):.4f} ، FPR = {fp}/{fp + tn} = {fp / (fp + tn):.4f}")

# المنحنى الكامل كما يحسبه sklearn
fpr, tpr, thresholds = roc_curve(y_true, scores)
print("العتبات التي فحصها:", thresholds)
print("FPR:", fpr.round(4))
print("TPR:", tpr.round(4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([1, 1, 0, 1, 0, 0])
scores = np.array([0.9, 0.6, 0.55, 0.4, 0.3, 0.2])

pos = scores[y_true == 1]
neg = scores[y_true == 0]
wins = sum(1 for p in pos for q in neg if p > q)
auc_manual = wins / (len(pos) * len(neg))

print(f"الأزواج الرابحة: {wins} من {len(pos) * len(neg)} أزواج")
print(f"AUC بالعد اليدوي: {auc_manual:.4f}")
print(f"AUC من roc_auc_score: {roc_auc_score(y_true, scores):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)

print(f"AUC متعدد الفئات (ovr, macro):  {roc_auc_score(y_te, proba, multi_class='ovr', average='macro'):.4f}")
print(f"AUC متعدد الفئات (ovr, micro):  {roc_auc_score(y_te, proba, multi_class='ovr', average='micro'):.4f}")
print(f"AUC متعدد الفئات (ovo, macro):  {roc_auc_score(y_te, proba, multi_class='ovo', average='macro'):.4f}")

# الخطأ الشائع: تمرير درجات decision_function بدل الاحتمالات
try:
    roc_auc_score(y_te, model.decision_function(X_te), multi_class='ovr')
except ValueError as err:
    print("ValueError:", str(err)[:100])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)

X, y = make_classification(
    n_samples=20000, n_features=12, n_informative=6, n_redundant=2,
    weights=[0.99, 0.01], class_sep=1.5, flip_y=0.0, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"نسبة الإيجابيات في الاختبار: {y_te.mean():.2%}")

log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
s_log = log.predict_proba(X_te)[:, 1]
hgb = HistGradientBoostingClassifier(max_iter=120, random_state=42).fit(X_tr, y_tr)
s_hgb = hgb.predict_proba(X_te)[:, 1]

print(f"اللوجيستي: AUC = {roc_auc_score(y_te, s_log):.4f} | AP = {average_precision_score(y_te, s_log):.4f}")
print(f"التدرّجي:  AUC = {roc_auc_score(y_te, s_hgb):.4f} | AP = {average_precision_score(y_te, s_hgb):.4f}")

# حقيقة التشغيل: الدقة الفعلية عند مستويات استرجاع نحتاجها
import numpy as np
from sklearn.metrics import precision_recall_curve

prec, rec, _ = precision_recall_curve(y_te, s_log)
order = np.argsort(rec)                  # الاسترجاع تنازلي: نرتّبه صعودياً للاستيفاء
for r in [0.5, 0.9]:
    p_at_r = float(np.interp(r, rec[order], prec[order]))
    print(f"اللوجيستي عند استرجاع {r:.0%}: الدقة الفعلية = {p_at_r:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
constant_scores = np.full(len(y_te), dummy.predict_proba(X_te)[0, 1])

print(f"معدل الانتشار: {y_te.mean():.4f}")
print(f"AUC للدرجات الثابتة: {roc_auc_score(y_te, constant_scores):.4f} (القطر)")
print(f"AP للدرجات الثابتة:  {average_precision_score(y_te, constant_scores):.4f} (خط أفقي)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import det_curve

fpr_l, fnr_l, _ = det_curve(y_te, s_log)
fpr_h, fnr_h, _ = det_curve(y_te, s_hgb)

order_l = np.argsort(fpr_l)
order_h = np.argsort(fpr_h)
for target_fpr in [0.01, 0.05]:
    fnr_at_l = float(np.interp(target_fpr, fpr_l[order_l], fnr_l[order_l]))
    fnr_at_h = float(np.interp(target_fpr, fpr_h[order_h], fnr_h[order_h]))
    print(f"عند FPR = {target_fpr:.0%}: FNR اللوجيستي = {fnr_at_l:.4f} | FNR التدرّجي = {fnr_at_h:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
scores = np.array([0.85, 0.7, 0.6, 0.5, 0.45, 0.3, 0.25, 0.1])

pos = scores[y_true == 1]
neg = scores[y_true == 0]
wins = sum(1 for p in pos for q in neg if p > q)

print(f"الأزواج الرابحة: {wins} من {len(pos) * len(neg)}")
print(f"AUC يدوياً: {wins / (len(pos) * len(neg)):.4f} | roc_auc_score: {roc_auc_score(y_true, scores):.4f}")
fpr, tpr, thr = roc_curve(y_true, scores)
print(f"عدد نقاط المنحنى: {len(fpr)} | عدد العتبات: {len(thr)}")
print("FPR:", fpr.round(4))
print("TPR:", tpr.round(4))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-76-ch76.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
random_scores = rng.random(2000)
y_random = (rng.random(2000) < 0.05).astype(int)   # انتشار 5%

print(f"الانتشار: {y_random.mean():.4f}")
print(f"AUC درجات عشوائية: {roc_auc_score(y_random, random_scores):.4f} (المفترض: 0.5)")
print(f"AP درجات عشوائية:  {average_precision_score(y_random, random_scores):.4f} (المفترض: الانتشار)")
