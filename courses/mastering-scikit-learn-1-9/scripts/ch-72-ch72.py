#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقسيم البيانات: الخطوة التي تُفشل المشاريع
Mastering scikit-learn 1.9 — Chapter 72 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-72-ch72 (lesson 72)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-72-ch72.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقسيم البيانات: الخطوة التي تُفشل المشاريع
Mastering scikit-learn 1.9 — Chapter 72 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-72-ch72 (lesson 72)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"الدقة على بيانات التدريب (رآها أثناء الملاءمة): {tree.score(X_tr, y_tr):.4f}")
print(f"الدقة على بيانات الاختبار (لم يرها قط):      {tree.score(X_te, y_te):.4f}")
print(f"الفجوة: {tree.score(X_tr, y_tr) - tree.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# (1) test_size كعدد صريح + stratify يوازن التوزيع
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
print(f"الأحجام: تدريب {len(X_tr)} / اختبار {len(X_te)}")
print(f"نسبة الموجبة: الكل {y.mean():.4f} | تدريب {y_tr.mean():.4f} | اختبار {y_te.mean():.4f}")

# (2) نفس البذرة = نفس التقسيم؛ بذرة مختلفة = تقسيم آخر
a = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
b = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
c = train_test_split(X, y, test_size=150, random_state=7, stratify=y)
print("نفس البذرة: تطابق؟", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
print("بذرتان مختلفتان: تطابق؟", np.array_equal(a[0], c[0]))

# (3) shuffle=False يحافظ على الترتيب
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
print("مع shuffle=False: التدريب أول الصفوف؟", np.array_equal(X_tr, X[:len(X_tr)]))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20,
                           weights=[0.95, 0.05], flip_y=0, random_state=0)
print(f"إجمالي الإيجابيات: {int(y.sum())} من {len(y)} (نسبة {y.mean():.3f})")

plain = [int(y_te.sum()) for s in range(20)
         for _, _, _, y_te in [train_test_split(X, y, test_size=0.2, random_state=s)]]
strat = [int(y_te.sum()) for s in range(20)
         for _, _, _, y_te in [train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)]]
print(f"إيجابيات الاختبار عبر 20 بذرة بلا stratify: {plain}")
print(f"أدنى {min(plain)} وأعلى {max(plain)} مقابل المتوقع {0.2 * int(y.sum()):.0f}")
print(f"مع stratify: {sorted(set(strat))} في كل البذور")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
scores = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_tr), y_tr)
    scores.append(model.score(scaler.transform(X_te), y_te))

scores = np.array(scores)
print(f"أدنى دقة: {scores.min():.4f} | أعلى دقة: {scores.max():.4f}")
print(f"المتوسط: {scores.mean():.4f} | الانحراف المعياري: {scores.std():.4f}")
print(f"المدى الكامل: {scores.max() - scores.min():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
X = rng.normal(size=(300, 2000))       # ضجيج نقي: لا علاقة بالتسميات
y = rng.integers(0, 2, 300)

# المسرب: تحجيم + انتقاء على كل البيانات قبل التقسيم
X_scaled = StandardScaler().fit_transform(X)
X_best = SelectKBest(f_classif, k=20).fit_transform(X_scaled, y)
Xtr, Xte, ytr, yte = train_test_split(X_best, y, test_size=0.5,
                                      random_state=42, stratify=y)
leaky = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(f"الدقة المزيفة (معالجة قبل التقسيم): {leaky.score(Xte, yte):.4f}")

# النزيه: نفس الخطوات، مُلائمة على التدريب فقط داخل Pipeline
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                      random_state=42, stratify=y)
honest = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20),
                       LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(f"الدقة الحقيقية (معالجة داخل Pipeline): {honest.score(Xte, yte):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(42)
n = 600
t = np.arange(n)
x = t / 60.0 + rng.normal(0, 0.05, n)     # إشارة تنزلق ببطء مع الزمن
y = 3.0 * x + rng.normal(0, 0.2, n)       # الهدف يتبعها
X = x.reshape(-1, 1)

for name, kw in [("shuffle=True (مخلوط)", dict(random_state=42)),
                 ("shuffle=False (تتابعي)", dict(shuffle=False))]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, **kw)
    model = KNeighborsRegressor(n_neighbors=5).fit(Xtr, ytr)
    print(f"{name}: R2 = {model.score(Xte, yte):+.4f} | تدريب [{Xtr.min():.2f}, {Xtr.max():.2f}] | اختبار [{Xte.min():.2f}, {Xte.max():.2f}]")
# مقدمة تسرب المجموعات: التقسيم على مستوى المجموعة لا الصف
groups = np.repeat(np.arange(30), 20)     # 30 'مريضاً' بكل منهم 20 زيارة
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, te_idx = next(gss.split(np.zeros((600, 2)), groups=groups))
overlap = set(np.unique(groups[tr_idx])) & set(np.unique(groups[te_idx]))
print(f"المجموعات: تدريب {len(np.unique(groups[tr_idx]))} | اختبار {len(np.unique(groups[te_idx]))} | تداخل: {len(overlap)}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_wine(return_X_y=True)
counts_plain, counts_strat = [], []
for seed in range(20):
    _, _, _, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    counts_plain.append(np.bincount(y_te, minlength=3))
    _, _, _, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    counts_strat.append(np.bincount(y_te, minlength=3))

plain, strat = np.array(counts_plain), np.array(counts_strat)
print(f"الفئة الأصغر (48 عينة) في الاختبار بلا stratify: من {plain[:, 2].min()} إلى {plain[:, 2].max()}")
print(f"مع stratify: {sorted(set(strat[:, 2].tolist()))} في كل التكرارات")
print(f"بذرة 3 بلا stratify: {plain[3].tolist()} | معها: {strat[3].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch72.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(42)
t = np.arange(600) / 60.0
x = t + 0.05 * rng.normal(size=600)
y = 3.0 * x + 2.0 * np.sin(3.0 * t) + 0.2 * rng.normal(size=600)

for name, kw in [("مخلوط  ", dict(random_state=42)),
                 ("تتابعي", dict(shuffle=False))]:
    Xtr, Xte, ytr, yte = train_test_split(x.reshape(-1, 1), y, test_size=0.3, **kw)
    for label, mdl in [("LinearRegression", LinearRegression()),
                       ("KNeighbors(k=5)  ", KNeighborsRegressor(n_neighbors=5))]:
        r2 = mdl.fit(Xtr, ytr).score(Xte, yte)
        print(f"{name} | {label}: R2 = {r2:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"الدقة على بيانات التدريب (رآها أثناء الملاءمة): {tree.score(X_tr, y_tr):.4f}")
print(f"الدقة على بيانات الاختبار (لم يرها قط):      {tree.score(X_te, y_te):.4f}")
print(f"الفجوة: {tree.score(X_tr, y_tr) - tree.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# (1) test_size كعدد صريح + stratify يوازن التوزيع
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
print(f"الأحجام: تدريب {len(X_tr)} / اختبار {len(X_te)}")
print(f"نسبة الموجبة: الكل {y.mean():.4f} | تدريب {y_tr.mean():.4f} | اختبار {y_te.mean():.4f}")

# (2) نفس البذرة = نفس التقسيم؛ بذرة مختلفة = تقسيم آخر
a = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
b = train_test_split(X, y, test_size=150, random_state=42, stratify=y)
c = train_test_split(X, y, test_size=150, random_state=7, stratify=y)
print("نفس البذرة: تطابق؟", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
print("بذرتان مختلفتان: تطابق؟", np.array_equal(a[0], c[0]))

# (3) shuffle=False يحافظ على الترتيب
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
print("مع shuffle=False: التدريب أول الصفوف؟", np.array_equal(X_tr, X[:len(X_tr)]))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20,
                           weights=[0.95, 0.05], flip_y=0, random_state=0)
print(f"إجمالي الإيجابيات: {int(y.sum())} من {len(y)} (نسبة {y.mean():.3f})")

plain = [int(y_te.sum()) for s in range(20)
         for _, _, _, y_te in [train_test_split(X, y, test_size=0.2, random_state=s)]]
strat = [int(y_te.sum()) for s in range(20)
         for _, _, _, y_te in [train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)]]
print(f"إيجابيات الاختبار عبر 20 بذرة بلا stratify: {plain}")
print(f"أدنى {min(plain)} وأعلى {max(plain)} مقابل المتوقع {0.2 * int(y.sum()):.0f}")
print(f"مع stratify: {sorted(set(strat))} في كل البذور")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
scores = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_tr), y_tr)
    scores.append(model.score(scaler.transform(X_te), y_te))

scores = np.array(scores)
print(f"أدنى دقة: {scores.min():.4f} | أعلى دقة: {scores.max():.4f}")
print(f"المتوسط: {scores.mean():.4f} | الانحراف المعياري: {scores.std():.4f}")
print(f"المدى الكامل: {scores.max() - scores.min():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
X = rng.normal(size=(300, 2000))       # ضجيج نقي: لا علاقة بالتسميات
y = rng.integers(0, 2, 300)

# المسرب: تحجيم + انتقاء على كل البيانات قبل التقسيم
X_scaled = StandardScaler().fit_transform(X)
X_best = SelectKBest(f_classif, k=20).fit_transform(X_scaled, y)
Xtr, Xte, ytr, yte = train_test_split(X_best, y, test_size=0.5,
                                      random_state=42, stratify=y)
leaky = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(f"الدقة المزيفة (معالجة قبل التقسيم): {leaky.score(Xte, yte):.4f}")

# النزيه: نفس الخطوات، مُلائمة على التدريب فقط داخل Pipeline
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                      random_state=42, stratify=y)
honest = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20),
                       LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(f"الدقة الحقيقية (معالجة داخل Pipeline): {honest.score(Xte, yte):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(42)
n = 600
t = np.arange(n)
x = t / 60.0 + rng.normal(0, 0.05, n)     # إشارة تنزلق ببطء مع الزمن
y = 3.0 * x + rng.normal(0, 0.2, n)       # الهدف يتبعها
X = x.reshape(-1, 1)

for name, kw in [("shuffle=True (مخلوط)", dict(random_state=42)),
                 ("shuffle=False (تتابعي)", dict(shuffle=False))]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, **kw)
    model = KNeighborsRegressor(n_neighbors=5).fit(Xtr, ytr)
    print(f"{name}: R2 = {model.score(Xte, yte):+.4f} | تدريب [{Xtr.min():.2f}, {Xtr.max():.2f}] | اختبار [{Xte.min():.2f}, {Xte.max():.2f}]")
# مقدمة تسرب المجموعات: التقسيم على مستوى المجموعة لا الصف
groups = np.repeat(np.arange(30), 20)     # 30 'مريضاً' بكل منهم 20 زيارة
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, te_idx = next(gss.split(np.zeros((600, 2)), groups=groups))
overlap = set(np.unique(groups[tr_idx])) & set(np.unique(groups[te_idx]))
print(f"المجموعات: تدريب {len(np.unique(groups[tr_idx]))} | اختبار {len(np.unique(groups[te_idx]))} | تداخل: {len(overlap)}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_wine(return_X_y=True)
counts_plain, counts_strat = [], []
for seed in range(20):
    _, _, _, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    counts_plain.append(np.bincount(y_te, minlength=3))
    _, _, _, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    counts_strat.append(np.bincount(y_te, minlength=3))

plain, strat = np.array(counts_plain), np.array(counts_strat)
print(f"الفئة الأصغر (48 عينة) في الاختبار بلا stratify: من {plain[:, 2].min()} إلى {plain[:, 2].max()}")
print(f"مع stratify: {sorted(set(strat[:, 2].tolist()))} في كل التكرارات")
print(f"بذرة 3 بلا stratify: {plain[3].tolist()} | معها: {strat[3].tolist()}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-72-ch72.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(42)
t = np.arange(600) / 60.0
x = t + 0.05 * rng.normal(size=600)
y = 3.0 * x + 2.0 * np.sin(3.0 * t) + 0.2 * rng.normal(size=600)

for name, kw in [("مخلوط  ", dict(random_state=42)),
                 ("تتابعي", dict(shuffle=False))]:
    Xtr, Xte, ytr, yte = train_test_split(x.reshape(-1, 1), y, test_size=0.3, **kw)
    for label, mdl in [("LinearRegression", LinearRegression()),
                       ("KNeighbors(k=5)  ", KNeighborsRegressor(n_neighbors=5))]:
        r2 = mdl.fit(Xtr, ytr).score(Xte, yte)
        print(f"{name} | {label}: R2 = {r2:+.4f}")
