#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار البيزي
Mastering scikit-learn 1.9 — Chapter 34 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-34-ch34 (lesson 34)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-34-ch34.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار البيزي
Mastering scikit-learn 1.9 — Chapter 34 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-34-ch34 (lesson 34)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch34.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# تدريب مقدر الحافة البيزي
bayes = BayesianRidge(compute_score=True)
bayes.fit(X_scaled, y)

# التنبؤ بالقيمة والانحراف المعياري لعدم اليقين
y_pred, y_std = bayes.predict(X_scaled[:5], return_std=True)

print("التنبؤات الخمس الأولى مع عدم اليقين المصاحب (±1σ):")
for i in range(5):
    print(f"  العينة {i+1}: التنبؤ = {y_pred[i]:.1f} ± {y_std[i]:.2f}")

print(f"\nمعلمة دقة الضجيج المقدرة ذاتياً (alpha_): {bayes.alpha_:.4f}")
print(f"معلمة تنظيم الأوزان المقدرة ذاتياً (lambda_): {bayes.lambda_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch34.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import BayesianRidge

rng = np.random.default_rng(42)
X_tr = np.sort(rng.uniform(-3, 3, size=(25, 1)), axis=0)
y_tr = np.sin(X_tr).ravel() + rng.normal(scale=0.15, size=25)

bayes = BayesianRidge().fit(X_tr, y_tr)

# التنبؤ على نطاق واسع يتجاوز فضاء التدريب
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
y_pred, y_std = bayes.predict(X_test, return_std=True)

# مقارنة عدم اليقين داخل نطاق التدريب وخارجه
std_inside = y_std[np.abs(X_test.ravel()) <= 3.0].mean()
std_outside = y_std[np.abs(X_test.ravel()) > 3.0].mean()

print(f"متوسط عدم اليقين داخل نطاق البيانات [-3, 3]:  {std_inside:.4f}")
print(f"متوسط عدم اليقين خارج نطاق البيانات (الاستقراء): {std_outside:.4f}")
print("النموذج البيزي يعبر بوضوح عن نقص يقينه كلما ابتعد عن بيانات التدريب!")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch34.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import ARDRegression
from sklearn.datasets import make_regression
import numpy as np

# بيانات تحوي 20 ميزة، 4 ميزات فقط حقيقية
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=4,
    noise=5.0, coef=True, random_state=42
)

ard = ARDRegression(compute_score=True).fit(X, y)

estimated_zero = np.isclose(ard.coef_, 0.0, atol=1e-2)
actual_zero = true_coef == 0.0

print(f"عدد الميزات الحقيقية غير المفيدة: {np.sum(actual_zero)}")
print(f"عدد الميزات التي صفرها ARD بنجاح: {np.sum(estimated_zero)}")
print(f"دقة تطابق تصفير الميزات غير المفيدة: {np.mean(estimated_zero == actual_zero):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch34.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1000], data.target[:1000], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

bayes = BayesianRidge().fit(X_tr_sc, y_tr)
y_pred, y_std = bayes.predict(X_te_sc, return_std=True)

errors = np.abs(y_te - y_pred)
corr = np.corrcoef(errors, y_std)[0, 1]

print(f"معامل الارتباط بين خطأ التنبؤ الفعلي والانحراف المعياري للشك: {corr:.3f}")
print("الارتباط الإيجابي يؤكد أن النموذج يعلم متى تزداد احتمالية خطئه!")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch34.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_sc = StandardScaler().fit_transform(X)

bayes = BayesianRidge(compute_score=True, max_iter=300).fit(X_sc, y)

print(f"عدد التكرارات حتى التقارب: {len(bayes.scores_)}")
print(f"قيمة دالة الاحتمال الهامشي في الخطوة الأولى: {bayes.scores_[0]:.2f}")
print(f"قيمة دالة الاحتمال الهامشي النهائية:       {bayes.scores_[-1]:.2f}")
print(f"التحسن الإجمالي في دالة الهدف: {bayes.scores_[-1] - bayes.scores_[0]:+.2f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-34-ch34.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# تدريب مقدر الحافة البيزي
bayes = BayesianRidge(compute_score=True)
bayes.fit(X_scaled, y)

# التنبؤ بالقيمة والانحراف المعياري لعدم اليقين
y_pred, y_std = bayes.predict(X_scaled[:5], return_std=True)

print("التنبؤات الخمس الأولى مع عدم اليقين المصاحب (±1σ):")
for i in range(5):
    print(f"  العينة {i+1}: التنبؤ = {y_pred[i]:.1f} ± {y_std[i]:.2f}")

print(f"\nمعلمة دقة الضجيج المقدرة ذاتياً (alpha_): {bayes.alpha_:.4f}")
print(f"معلمة تنظيم الأوزان المقدرة ذاتياً (lambda_): {bayes.lambda_:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-34-ch34.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import BayesianRidge

rng = np.random.default_rng(42)
X_tr = np.sort(rng.uniform(-3, 3, size=(25, 1)), axis=0)
y_tr = np.sin(X_tr).ravel() + rng.normal(scale=0.15, size=25)

bayes = BayesianRidge().fit(X_tr, y_tr)

# التنبؤ على نطاق واسع يتجاوز فضاء التدريب
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
y_pred, y_std = bayes.predict(X_test, return_std=True)

# مقارنة عدم اليقين داخل نطاق التدريب وخارجه
std_inside = y_std[np.abs(X_test.ravel()) <= 3.0].mean()
std_outside = y_std[np.abs(X_test.ravel()) > 3.0].mean()

print(f"متوسط عدم اليقين داخل نطاق البيانات [-3, 3]:  {std_inside:.4f}")
print(f"متوسط عدم اليقين خارج نطاق البيانات (الاستقراء): {std_outside:.4f}")
print("النموذج البيزي يعبر بوضوح عن نقص يقينه كلما ابتعد عن بيانات التدريب!")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-34-ch34.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import ARDRegression
from sklearn.datasets import make_regression
import numpy as np

# بيانات تحوي 20 ميزة، 4 ميزات فقط حقيقية
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=4,
    noise=5.0, coef=True, random_state=42
)

ard = ARDRegression(compute_score=True).fit(X, y)

estimated_zero = np.isclose(ard.coef_, 0.0, atol=1e-2)
actual_zero = true_coef == 0.0

print(f"عدد الميزات الحقيقية غير المفيدة: {np.sum(actual_zero)}")
print(f"عدد الميزات التي صفرها ARD بنجاح: {np.sum(estimated_zero)}")
print(f"دقة تطابق تصفير الميزات غير المفيدة: {np.mean(estimated_zero == actual_zero):.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-34-ch34.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1000], data.target[:1000], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

bayes = BayesianRidge().fit(X_tr_sc, y_tr)
y_pred, y_std = bayes.predict(X_te_sc, return_std=True)

errors = np.abs(y_te - y_pred)
corr = np.corrcoef(errors, y_std)[0, 1]

print(f"معامل الارتباط بين خطأ التنبؤ الفعلي والانحراف المعياري للشك: {corr:.3f}")
print("الارتباط الإيجابي يؤكد أن النموذج يعلم متى تزداد احتمالية خطئه!")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-34-ch34.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_sc = StandardScaler().fit_transform(X)

bayes = BayesianRidge(compute_score=True, max_iter=300).fit(X_sc, y)

print(f"عدد التكرارات حتى التقارب: {len(bayes.scores_)}")
print(f"قيمة دالة الاحتمال الهامشي في الخطوة الأولى: {bayes.scores_[0]:.2f}")
print(f"قيمة دالة الاحتمال الهامشي النهائية:       {bayes.scores_[-1]:.2f}")
print(f"التحسن الإجمالي في دالة الهدف: {bayes.scores_[-1] - bayes.scores_[0]:+.2f}")
