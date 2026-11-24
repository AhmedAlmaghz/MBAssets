#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — بيز الساذج
Mastering scikit-learn 1.9 — Chapter 56 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-56-ch56 (lesson 56)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-56-ch56.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — بيز الساذج
Mastering scikit-learn 1.9 — Chapter 56 — Supervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-56-ch56 (lesson 56)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
# أرشيف: 100 رسالة مزعجة و400 عادية
prior_spam, prior_ham = 100 / 500, 400 / 500

# إحصاءات الأرشيف: احتمال الكلمتين في كل فئة
p_magan_free_spam, p_magan_free_ham = 0.60, 0.03   # «مجاني»
p_ard_spam, p_ard_ham = 0.50, 0.10                 # «عرض»

# الفرضية الساذجة: الكلمتان مستقلتان شرطاً داخل كل فئة
score_spam = prior_spam * p_magan_free_spam * p_ard_spam
score_ham = prior_ham * p_magan_free_ham * p_ard_ham

print(f"ناتج بيز للفئة المزعجة: {score_spam:.5f} | للفئة العادية: {score_ham:.5f}")
posterior = score_spam / (score_spam + score_ham)
print(f"الاحتمال اللاحق أن الرسالة مزعجة: {posterior:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

gnb = GaussianNB().fit(X_tr, y_tr)
print(f"دقة GaussianNB على الاختبار: {gnb.score(X_te, y_te):.4f}")

for vs in [1e-9, 1e-4, 1e-1]:
    score = cross_val_score(GaussianNB(var_smoothing=vs), X, y, cv=5).mean()
    print(f"var_smoothing = {vs:6.0e} | متوسط دقة التحقق الخماسي: {score:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam = ['عرض مجاني اربح الآن بسرعة', 'فزت جائزة مجانية اضغط الآن',
        'قرض مجاني بلا فوايد اربح اليوم', 'عرض خاص ساعات مجانية اليوم',
        'اربح المال الآن من المنزل', 'جائزة فورية اضغط هنا']
ham = ['اجتماع الفريق غدا في التاسعة', 'أرسل لي تقرير المشروع من فضلك',
       'موعد الطبيب يوم الثلاثاء', 'تقرير المبيعات جاهز للمراجعة',
       'شكرا على ملاحظاتك على الوثيقة', 'سافر الجمعة وأعود الأحد']

docs = spam + ham
y = np.array([1] * len(spam) + [0] * len(ham))
vec = CountVectorizer()
X = vec.fit_transform(docs)
print(f"مصفوفة العدّاد: {X.shape[0]} مستند و {X.shape[1]} كلمة، عناصر فعالة: {X.nnz}")

for a in [0.01, 0.1, 1.0]:
    m = MultinomialNB(alpha=a).fit(X, y)
    print(f"alpha = {a:5.2f} | دقة التدريب: {m.score(X, y):.4f}")

mnb = MultinomialNB(alpha=0.1).fit(X, y)
print(f"تنبؤ رسائل جديدة: {mnb.predict(vec.transform(['اربح جائزة مجانية الآن', 'تقرير الاجتماع غدا']))}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB

rng = np.random.default_rng(42)
n_maj, n_min = 900, 100
# فئة مهيمنة: كلمات عامة بكثافة متفاوتة
X_maj = rng.poisson(lam=4.0, size=(n_maj, 30)) * (rng.uniform(size=(n_maj, 30)) < 0.3)
# فئة نادرة: نفس الكلمات تقريباً مع تحيّز طفيف
X_min = rng.poisson(lam=1.5, size=(n_min, 30)) * (rng.uniform(size=(n_min, 30)) < 0.6) \
        + (rng.uniform(size=(n_min, 30)) < 0.2) * 3
Xb = np.vstack([X_maj, X_min]).astype(float)
yb = np.array([0] * n_maj + [1] * n_min)
X_tr, X_te, y_tr, y_te = train_test_split(Xb, yb, test_size=0.3, random_state=42, stratify=yb)

for name, model in [("MultinomialNB", MultinomialNB()), ("ComplementNB", ComplementNB())]:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"{name:13s} | الدقة الإجمالية: {(pred == y_te).mean():.4f} | "
          f"استرجاع الفئة النادرة: {pred[y_te == 1].mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
import numpy as np

# نفس مرسلات القسم السابق محفوظة في X و y — نثنّيها بعتبة 0.5
bnb = BernoulliNB(binarize=0.5).fit(X, y)
print(f"دقة BernoulliNB على المرسلات: {bnb.score(X, y):.4f}")

rng2 = np.random.default_rng(42)
Xf = rng2.integers(0, 3, size=(200, 4))
yf = Xf[:, 0] % 2  # الفئة تحددها قيمة الخاصية الأولى فقط
cnb = CategoricalNB().fit(Xf, yf)
print(f"دقة CategoricalNB على الفئوي: {cnb.score(Xf, yf):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=12000, n_features=8, random_state=42)
classes = np.unique(y)

stream_model = GaussianNB()
for start in range(0, 12000, 2000):
    stream_model.partial_fit(X[start:start + 2000], y[start:start + 2000], classes=classes)

full_model = GaussianNB().fit(X, y)
print(f"دقة التدريب بالدفعات: {stream_model.score(X, y):.4f}")
print(f"دقة التدريب الكامل:   {full_model.score(X, y):.4f}")
print(f"تطابق المتوسطات المقدَّرة: {np.allclose(stream_model.theta_, full_model.theta_)}")

lp = full_model.predict_log_proba(X[:3])
print(f"أول صف من predict_log_proba: {lp[0].round(4)}")
print(f"مجموع exp لكل صف: {np.exp(lp).sum(axis=1).round(4)}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)
score_nb = cross_val_score(GaussianNB(), X, y, cv=5).mean()
score_knn = cross_val_score(
    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), X, y, cv=5
).mean()
print(f"GaussianNB (بلا أي ضبط أو تطبيع): {score_nb:.4f}")
print(f"k-NN (بعد StandardScaler):        {score_knn:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part07/ch56.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=30000, n_features=12, random_state=42)
classes = np.unique(y)

stream = GaussianNB()
for start in range(0, 30000, 5000):
    stream.partial_fit(X[start:start + 5000], y[start:start + 5000], classes=classes)

full = GaussianNB().fit(X, y)
print(f"دقة الدفعات: {stream.score(X, y):.4f} | دقة الكامل: {full.score(X, y):.4f}")
print(f"تطابق المتوسطات: {np.allclose(stream.theta_, full.theta_)}")
print(f"تطابق التباينات: {np.allclose(stream.var_, full.var_)}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# أرشيف: 100 رسالة مزعجة و400 عادية
prior_spam, prior_ham = 100 / 500, 400 / 500

# إحصاءات الأرشيف: احتمال الكلمتين في كل فئة
p_magan_free_spam, p_magan_free_ham = 0.60, 0.03   # «مجاني»
p_ard_spam, p_ard_ham = 0.50, 0.10                 # «عرض»

# الفرضية الساذجة: الكلمتان مستقلتان شرطاً داخل كل فئة
score_spam = prior_spam * p_magan_free_spam * p_ard_spam
score_ham = prior_ham * p_magan_free_ham * p_ard_ham

print(f"ناتج بيز للفئة المزعجة: {score_spam:.5f} | للفئة العادية: {score_ham:.5f}")
posterior = score_spam / (score_spam + score_ham)
print(f"الاحتمال اللاحق أن الرسالة مزعجة: {posterior:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

gnb = GaussianNB().fit(X_tr, y_tr)
print(f"دقة GaussianNB على الاختبار: {gnb.score(X_te, y_te):.4f}")

for vs in [1e-9, 1e-4, 1e-1]:
    score = cross_val_score(GaussianNB(var_smoothing=vs), X, y, cv=5).mean()
    print(f"var_smoothing = {vs:6.0e} | متوسط دقة التحقق الخماسي: {score:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam = ['عرض مجاني اربح الآن بسرعة', 'فزت جائزة مجانية اضغط الآن',
        'قرض مجاني بلا فوايد اربح اليوم', 'عرض خاص ساعات مجانية اليوم',
        'اربح المال الآن من المنزل', 'جائزة فورية اضغط هنا']
ham = ['اجتماع الفريق غدا في التاسعة', 'أرسل لي تقرير المشروع من فضلك',
       'موعد الطبيب يوم الثلاثاء', 'تقرير المبيعات جاهز للمراجعة',
       'شكرا على ملاحظاتك على الوثيقة', 'سافر الجمعة وأعود الأحد']

docs = spam + ham
y = np.array([1] * len(spam) + [0] * len(ham))
vec = CountVectorizer()
X = vec.fit_transform(docs)
print(f"مصفوفة العدّاد: {X.shape[0]} مستند و {X.shape[1]} كلمة، عناصر فعالة: {X.nnz}")

for a in [0.01, 0.1, 1.0]:
    m = MultinomialNB(alpha=a).fit(X, y)
    print(f"alpha = {a:5.2f} | دقة التدريب: {m.score(X, y):.4f}")

mnb = MultinomialNB(alpha=0.1).fit(X, y)
print(f"تنبؤ رسائل جديدة: {mnb.predict(vec.transform(['اربح جائزة مجانية الآن', 'تقرير الاجتماع غدا']))}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB

rng = np.random.default_rng(42)
n_maj, n_min = 900, 100
# فئة مهيمنة: كلمات عامة بكثافة متفاوتة
X_maj = rng.poisson(lam=4.0, size=(n_maj, 30)) * (rng.uniform(size=(n_maj, 30)) < 0.3)
# فئة نادرة: نفس الكلمات تقريباً مع تحيّز طفيف
X_min = rng.poisson(lam=1.5, size=(n_min, 30)) * (rng.uniform(size=(n_min, 30)) < 0.6) \
        + (rng.uniform(size=(n_min, 30)) < 0.2) * 3
Xb = np.vstack([X_maj, X_min]).astype(float)
yb = np.array([0] * n_maj + [1] * n_min)
X_tr, X_te, y_tr, y_te = train_test_split(Xb, yb, test_size=0.3, random_state=42, stratify=yb)

for name, model in [("MultinomialNB", MultinomialNB()), ("ComplementNB", ComplementNB())]:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"{name:13s} | الدقة الإجمالية: {(pred == y_te).mean():.4f} | "
          f"استرجاع الفئة النادرة: {pred[y_te == 1].mean():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
import numpy as np

# نفس مرسلات القسم السابق محفوظة في X و y — نثنّيها بعتبة 0.5
bnb = BernoulliNB(binarize=0.5).fit(X, y)
print(f"دقة BernoulliNB على المرسلات: {bnb.score(X, y):.4f}")

rng2 = np.random.default_rng(42)
Xf = rng2.integers(0, 3, size=(200, 4))
yf = Xf[:, 0] % 2  # الفئة تحددها قيمة الخاصية الأولى فقط
cnb = CategoricalNB().fit(Xf, yf)
print(f"دقة CategoricalNB على الفئوي: {cnb.score(Xf, yf):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=12000, n_features=8, random_state=42)
classes = np.unique(y)

stream_model = GaussianNB()
for start in range(0, 12000, 2000):
    stream_model.partial_fit(X[start:start + 2000], y[start:start + 2000], classes=classes)

full_model = GaussianNB().fit(X, y)
print(f"دقة التدريب بالدفعات: {stream_model.score(X, y):.4f}")
print(f"دقة التدريب الكامل:   {full_model.score(X, y):.4f}")
print(f"تطابق المتوسطات المقدَّرة: {np.allclose(stream_model.theta_, full_model.theta_)}")

lp = full_model.predict_log_proba(X[:3])
print(f"أول صف من predict_log_proba: {lp[0].round(4)}")
print(f"مجموع exp لكل صف: {np.exp(lp).sum(axis=1).round(4)}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)
score_nb = cross_val_score(GaussianNB(), X, y, cv=5).mean()
score_knn = cross_val_score(
    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), X, y, cv=5
).mean()
print(f"GaussianNB (بلا أي ضبط أو تطبيع): {score_nb:.4f}")
print(f"k-NN (بعد StandardScaler):        {score_knn:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-56-ch56.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=30000, n_features=12, random_state=42)
classes = np.unique(y)

stream = GaussianNB()
for start in range(0, 30000, 5000):
    stream.partial_fit(X[start:start + 5000], y[start:start + 5000], classes=classes)

full = GaussianNB().fit(X, y)
print(f"دقة الدفعات: {stream.score(X, y):.4f} | دقة الكامل: {full.score(X, y):.4f}")
print(f"تطابق المتوسطات: {np.allclose(stream.theta_, full.theta_)}")
print(f"تطابق التباينات: {np.allclose(stream.var_, full.var_)}")
