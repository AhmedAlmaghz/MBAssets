#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس متقدّمة ومجموعات غير مستوية
Mastering scikit-learn 1.9 — Chapter 79 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-79-ch79 (lesson 79)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-79-ch79.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس متقدّمة ومجموعات غير مستوية
Mastering scikit-learn 1.9 — Chapter 79 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-79-ch79 (lesson 79)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (accuracy_score, hamming_loss, jaccard_score, f1_score)

y_true = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])

print(f"subset accuracy = {accuracy_score(y_true, y_pred):.4f}  (عينة واحدة صابت كل أبوابها)")
print(f"hamming_loss    = {hamming_loss(y_true, y_pred):.4f}  ({int(hamming_loss(y_true, y_pred) * 9)} باب خاطئ من 9)")

manual = [np.sum(t & p) / np.sum(t | p) for t, p in zip(y_true, y_pred)]
print(f"jaccard يدوياً لكل عينة: {np.round(manual, 2)} → متوسط = {np.mean(manual):.4f}")
print(f"jaccard المكتبة       : {jaccard_score(y_true, y_pred, average='samples'):.4f}")

print(f"F1 macro   = {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}  (وزن متساوٍ لكل باب)")
print(f"F1 samples = {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}  (وزن متساوٍ لكل عينة)")
print(f"F1 micro   = {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}  (تجميع الأعداد الكلية)")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import ndcg_score, dcg_score

rel    = np.array([[3., 2., 3., 0., 1., 2.]])   # الصلة الحقيقية لكل بند
scores = np.array([[0.90, 0.80, 0.50, 0.85, 0.30, 0.60]])  # تنبؤ النموذج

order  = rel[0][np.argsort(-scores[0])]          # ترتيب النموذج
ideal  = np.sort(rel[0])[::-1]                   # القائمة المثالية
disc   = 1 / np.log2(np.arange(1, 7) + 1)        # عامل الخصم لكل مرتبة
print(f"ترتيب النموذج   : {order}")
print(f"القائمة المثالية: {ideal}")
print(f"عوامل الخصم     : {np.round(disc, 3)}")

dcg_pred = np.sum((2**order - 1) * disc)
dcg_ideal = np.sum((2**ideal - 1) * disc)
print(f"DCG المتنبئ = {dcg_pred:.4f} مقابل IDCG = {dcg_ideal:.4f}")
print(f"NDCG = {ndcg_score(rel, scores):.4f}  |  DCG غير المطبَّعة = {dcg_score(rel, scores):.4f}")
print(f"NDCG@3 (قصّ عند 3 بنود كواجهة التطبيق) = {ndcg_score(rel, scores, k=3):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (label_ranking_average_precision_score,
    label_ranking_loss, coverage_error)

y_true  = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
y_score = np.array([[0.90, 0.10, 0.80, 0.30],
                    [0.20, 0.90, 0.70, 0.10],
                    [0.30, 0.85, 0.60, 0.20]])

print(f"LRAP            = {label_ranking_average_precision_score(y_true, y_score):.4f}")
print(f"ranking_loss    = {label_ranking_loss(y_true, y_score):.4f}  (زوج مقلوب واحد من 12)")
print(f"coverage_error  = {coverage_error(y_true, y_score):.4f}  (متوسط عمق أول صحيح)")

# العينة الثالثة يدوياً: ترتيب الدرجات [1(0.85)، 2(0.60)، 0(0.30)، 3(0.20)]
# التسميات الصحيحة {0, 1}: الأولى صحيحة (دقة 1/1)، والثانية عند العمق 3 (دقة 2/3)
print("دقة العينة الثالثة يدوياً: (1/1 + 2/3) / 2 =", (1.0 + 2/3) / 2)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score

X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                           class_sep=0.6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

for name, model in [("لوجستي", LogisticRegression(max_iter=1000)),
                    ("شجرة عميقة", DecisionTreeClassifier(random_state=42))]:
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"{name}: دقة القرار = {accuracy_score(y_te, proba >= 0.5):.4f} | "
          f"Brier = {brier_score_loss(y_te, proba):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, matthews_corrcoef)

X, y = make_classification(n_samples=1500, n_classes=3, weights=[0.80, 0.15, 0.05],
                           n_features=10, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
print(f"أعداد الفئات في الاختبار: {np.bincount(y_te)}")

for name, model in [("عادي", LogisticRegression(max_iter=1000)),
                    ("فئات موزونة", LogisticRegression(max_iter=1000, class_weight="balanced"))]:
    p = model.fit(X_tr, y_tr).predict(X_te)
    print(f"{name}: دقة = {accuracy_score(y_te, p):.4f} | "
          f"متوازنة = {balanced_accuracy_score(y_te, p):.4f} | "
          f"MCC = {matthews_corrcoef(y_te, p):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (top_k_accuracy_score, matthews_corrcoef,
    balanced_accuracy_score)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)
pred = model.predict(X_te)

print(f"top-1 = {top_k_accuracy_score(y_te, proba, k=1):.4f}")
print(f"top-3 = {top_k_accuracy_score(y_te, proba, k=3):.4f}")
print(f"balanced accuracy = {balanced_accuracy_score(y_te, pred):.4f}")
print(f"MCC (عشر فئات) = {matthews_corrcoef(y_te, pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, hamming_loss, jaccard_score, f1_score)

X, Y = make_multilabel_classification(n_samples=600, n_features=12,
                                      n_classes=4, n_labels=2, random_state=42)
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.3, random_state=42)
P = KNeighborsClassifier(n_neighbors=5).fit(X_tr, Y_tr).predict(X_te)

print(f"subset accuracy (الأشد): {accuracy_score(Y_te, P):.4f}")
print(f"jaccard samples        : {jaccard_score(Y_te, P, average='samples', zero_division=0):.4f}")
print(f"F1 samples             : {f1_score(Y_te, P, average='samples', zero_division=0):.4f}")
print(f"hamming_loss (الألطف)  : {hamming_loss(Y_te, P):.4f}")
print("الترتيب متوقع: subset يشترط صواب كل الأبواب، و hamming يغتفر الأغلبية")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch79.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                           class_sep=0.6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

models = [("dummy (أعمى)", DummyClassifier(strategy="prior")),
          ("لوجستي", LogisticRegression(max_iter=1000)),
          ("شجرة عميقة", DecisionTreeClassifier(random_state=42))]
for name, m in models:
    proba = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    print(f"{name}: Brier = {brier_score_loss(y_te, proba):.4f}")
print("الأعمى حصل على 0.2500 بالضبط — ولا نموذج يتجاوزه حقاً إلا بتعلم حقيقي")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (accuracy_score, hamming_loss, jaccard_score, f1_score)

y_true = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])

print(f"subset accuracy = {accuracy_score(y_true, y_pred):.4f}  (عينة واحدة صابت كل أبوابها)")
print(f"hamming_loss    = {hamming_loss(y_true, y_pred):.4f}  ({int(hamming_loss(y_true, y_pred) * 9)} باب خاطئ من 9)")

manual = [np.sum(t & p) / np.sum(t | p) for t, p in zip(y_true, y_pred)]
print(f"jaccard يدوياً لكل عينة: {np.round(manual, 2)} → متوسط = {np.mean(manual):.4f}")
print(f"jaccard المكتبة       : {jaccard_score(y_true, y_pred, average='samples'):.4f}")

print(f"F1 macro   = {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}  (وزن متساوٍ لكل باب)")
print(f"F1 samples = {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}  (وزن متساوٍ لكل عينة)")
print(f"F1 micro   = {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}  (تجميع الأعداد الكلية)")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import ndcg_score, dcg_score

rel    = np.array([[3., 2., 3., 0., 1., 2.]])   # الصلة الحقيقية لكل بند
scores = np.array([[0.90, 0.80, 0.50, 0.85, 0.30, 0.60]])  # تنبؤ النموذج

order  = rel[0][np.argsort(-scores[0])]          # ترتيب النموذج
ideal  = np.sort(rel[0])[::-1]                   # القائمة المثالية
disc   = 1 / np.log2(np.arange(1, 7) + 1)        # عامل الخصم لكل مرتبة
print(f"ترتيب النموذج   : {order}")
print(f"القائمة المثالية: {ideal}")
print(f"عوامل الخصم     : {np.round(disc, 3)}")

dcg_pred = np.sum((2**order - 1) * disc)
dcg_ideal = np.sum((2**ideal - 1) * disc)
print(f"DCG المتنبئ = {dcg_pred:.4f} مقابل IDCG = {dcg_ideal:.4f}")
print(f"NDCG = {ndcg_score(rel, scores):.4f}  |  DCG غير المطبَّعة = {dcg_score(rel, scores):.4f}")
print(f"NDCG@3 (قصّ عند 3 بنود كواجهة التطبيق) = {ndcg_score(rel, scores, k=3):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (label_ranking_average_precision_score,
    label_ranking_loss, coverage_error)

y_true  = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
y_score = np.array([[0.90, 0.10, 0.80, 0.30],
                    [0.20, 0.90, 0.70, 0.10],
                    [0.30, 0.85, 0.60, 0.20]])

print(f"LRAP            = {label_ranking_average_precision_score(y_true, y_score):.4f}")
print(f"ranking_loss    = {label_ranking_loss(y_true, y_score):.4f}  (زوج مقلوب واحد من 12)")
print(f"coverage_error  = {coverage_error(y_true, y_score):.4f}  (متوسط عمق أول صحيح)")

# العينة الثالثة يدوياً: ترتيب الدرجات [1(0.85)، 2(0.60)، 0(0.30)، 3(0.20)]
# التسميات الصحيحة {0, 1}: الأولى صحيحة (دقة 1/1)، والثانية عند العمق 3 (دقة 2/3)
print("دقة العينة الثالثة يدوياً: (1/1 + 2/3) / 2 =", (1.0 + 2/3) / 2)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score

X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                           class_sep=0.6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

for name, model in [("لوجستي", LogisticRegression(max_iter=1000)),
                    ("شجرة عميقة", DecisionTreeClassifier(random_state=42))]:
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"{name}: دقة القرار = {accuracy_score(y_te, proba >= 0.5):.4f} | "
          f"Brier = {brier_score_loss(y_te, proba):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, matthews_corrcoef)

X, y = make_classification(n_samples=1500, n_classes=3, weights=[0.80, 0.15, 0.05],
                           n_features=10, n_informative=6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
print(f"أعداد الفئات في الاختبار: {np.bincount(y_te)}")

for name, model in [("عادي", LogisticRegression(max_iter=1000)),
                    ("فئات موزونة", LogisticRegression(max_iter=1000, class_weight="balanced"))]:
    p = model.fit(X_tr, y_tr).predict(X_te)
    print(f"{name}: دقة = {accuracy_score(y_te, p):.4f} | "
          f"متوازنة = {balanced_accuracy_score(y_te, p):.4f} | "
          f"MCC = {matthews_corrcoef(y_te, p):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (top_k_accuracy_score, matthews_corrcoef,
    balanced_accuracy_score)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)
pred = model.predict(X_te)

print(f"top-1 = {top_k_accuracy_score(y_te, proba, k=1):.4f}")
print(f"top-3 = {top_k_accuracy_score(y_te, proba, k=3):.4f}")
print(f"balanced accuracy = {balanced_accuracy_score(y_te, pred):.4f}")
print(f"MCC (عشر فئات) = {matthews_corrcoef(y_te, pred):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, hamming_loss, jaccard_score, f1_score)

X, Y = make_multilabel_classification(n_samples=600, n_features=12,
                                      n_classes=4, n_labels=2, random_state=42)
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.3, random_state=42)
P = KNeighborsClassifier(n_neighbors=5).fit(X_tr, Y_tr).predict(X_te)

print(f"subset accuracy (الأشد): {accuracy_score(Y_te, P):.4f}")
print(f"jaccard samples        : {jaccard_score(Y_te, P, average='samples', zero_division=0):.4f}")
print(f"F1 samples             : {f1_score(Y_te, P, average='samples', zero_division=0):.4f}")
print(f"hamming_loss (الألطف)  : {hamming_loss(Y_te, P):.4f}")
print("الترتيب متوقع: subset يشترط صواب كل الأبواب، و hamming يغتفر الأغلبية")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-79-ch79.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                           class_sep=0.6, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

models = [("dummy (أعمى)", DummyClassifier(strategy="prior")),
          ("لوجستي", LogisticRegression(max_iter=1000)),
          ("شجرة عميقة", DecisionTreeClassifier(random_state=42))]
for name, m in models:
    proba = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    print(f"{name}: Brier = {brier_score_loss(y_te, proba):.4f}")
print("الأعمى حصل على 0.2500 بالضبط — ولا نموذج يتجاوزه حقاً إلا بتعلم حقيقي")
