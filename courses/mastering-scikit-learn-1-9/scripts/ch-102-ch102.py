#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٤: التعرّف على الأرقام المكتوبة بخط اليد
Mastering scikit-learn 1.9 — Chapter 102 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-102-ch102 (lesson 102)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-102-ch102.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٤: التعرّف على الأرقام المكتوبة بخط اليد
Mastering scikit-learn 1.9 — Chapter 102 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-102-ch102 (lesson 102)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
print("صور:", digits.images.shape)
print("خصائص مسطحة:", digits.data.shape)
print("الفئات:", np.unique(digits.target).tolist())
print("مدى البكسل:", int(digits.images.min()), int(digits.images.max()))
print("توزيع الفئات:", np.bincount(digits.target).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits = load_digits()
Xtr, Xte, ytr, yte = train_test_split(
    digits.data, digits.target, test_size=0.25,
    stratify=digits.target, random_state=42)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("حجم التدريب والاختبار:", Xtr.shape[0], Xte.shape[0])
print("دقة الاختبار:", round(accuracy_score(yte, pred), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

digits = load_digits()
a, b, c, d = train_test_split(
    digits.data, digits.target, stratify=digits.target,
    test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000)).fit(a, c)
pred = model.predict(b)
print(classification_report(d, pred, digits=3, zero_division=0))
cm = confusion_matrix(d, pred)
print("شكل المصفوفة:", cm.shape)
print("أكثر أزواج الالتباس خارج القطر:")
for true_label, pred_label in sorted(
        ((i, j) for i in range(10) for j in range(10) if i != j),
        key=lambda pair: cm[pair[0], pair[1]], reverse=True)[:5]:
    print(f"{true_label} -> {pred_label}: {cm[true_label, pred_label]}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

digits = load_digits()
a, b, c, d = train_test_split(
    digits.data, digits.target, stratify=digits.target,
    test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(), SVC(C=3, gamma="scale"))
model.fit(a, c)
image = digits.images[0]
flat = image.reshape(1, -1)
print("شكل الصورة:", image.shape)
print("الرقم الحقيقي:", int(digits.target[0]))
print("التنبؤ:", int(model.predict(flat)[0]))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits

image = load_digits().images[0]
flat = image.reshape(-1)
print("الشكل:", image.shape, "عدد الخصائص:", flat.size)
assert flat.size == 64

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch102.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = load_digits()
a, b, c, d = train_test_split(
    D.data, D.target, stratify=D.target, random_state=42)
models = [
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(3))),
    ("logistic", make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=3000))),
]
for name, model in models:
    pred = model.fit(a, c).predict(b)
    print(name)
    print(classification_report(d, pred, digits=2, zero_division=0))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
print("صور:", digits.images.shape)
print("خصائص مسطحة:", digits.data.shape)
print("الفئات:", np.unique(digits.target).tolist())
print("مدى البكسل:", int(digits.images.min()), int(digits.images.max()))
print("توزيع الفئات:", np.bincount(digits.target).tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits = load_digits()
Xtr, Xte, ytr, yte = train_test_split(
    digits.data, digits.target, test_size=0.25,
    stratify=digits.target, random_state=42)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("حجم التدريب والاختبار:", Xtr.shape[0], Xte.shape[0])
print("دقة الاختبار:", round(accuracy_score(yte, pred), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

digits = load_digits()
a, b, c, d = train_test_split(
    digits.data, digits.target, stratify=digits.target,
    test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000)).fit(a, c)
pred = model.predict(b)
print(classification_report(d, pred, digits=3, zero_division=0))
cm = confusion_matrix(d, pred)
print("شكل المصفوفة:", cm.shape)
print("أكثر أزواج الالتباس خارج القطر:")
for true_label, pred_label in sorted(
        ((i, j) for i in range(10) for j in range(10) if i != j),
        key=lambda pair: cm[pair[0], pair[1]], reverse=True)[:5]:
    print(f"{true_label} -> {pred_label}: {cm[true_label, pred_label]}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

digits = load_digits()
a, b, c, d = train_test_split(
    digits.data, digits.target, stratify=digits.target,
    test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(), SVC(C=3, gamma="scale"))
model.fit(a, c)
image = digits.images[0]
flat = image.reshape(1, -1)
print("شكل الصورة:", image.shape)
print("الرقم الحقيقي:", int(digits.target[0]))
print("التنبؤ:", int(model.predict(flat)[0]))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits

image = load_digits().images[0]
flat = image.reshape(-1)
print("الشكل:", image.shape, "عدد الخصائص:", flat.size)
assert flat.size == 64

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-102-ch102.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = load_digits()
a, b, c, d = train_test_split(
    D.data, D.target, stratify=D.target, random_state=42)
models = [
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(3))),
    ("logistic", make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=3000))),
]
for name, model in models:
    pred = model.fit(a, c).predict(b)
    print(name)
    print(classification_report(d, pred, digits=2, zero_division=0))
