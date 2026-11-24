#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار حسب التوزيع
Mastering scikit-learn 1.9 — Chapter 33 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-33-ch33 (lesson 33)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-33-ch33.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الانحدار حسب التوزيع
Mastering scikit-learn 1.9 — Chapter 33 — Linear

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-33-ch33 (lesson 33)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part04/ch33.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.metrics import mean_poisson_deviance

rng = np.random.default_rng(42)
n_samples = 250
X = rng.uniform(0, 5, size=(n_samples, 2))

# معدل الحدوث الحقيقي يتبع دالة أسية (Poisson)
true_lambda = np.exp(0.5 + 0.6 * X[:, 0] - 0.4 * X[:, 1])
y = rng.poisson(lam=true_lambda)

ols = LinearRegression().fit(X, y)
poisson = PoissonRegressor(alpha=1e-3, max_iter=300).fit(X, y)

print(f"أدنى قيمة تنبأ بها OLS:     {ols.predict(X).min():.3f} (قد تكون سالبة!)")
print(f"أدنى قيمة تنبأ بها Poisson: {poisson.predict(X).min():.3f} (موجبة قطعاً)")
print(f"انحراف بواسون لـ OLS:       {mean_poisson_deviance(y, np.clip(ols.predict(X), 1e-6, None)):.4f}")
print(f"انحراف بواسون لـ Poisson:   {mean_poisson_deviance(y, poisson.predict(X)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part04/ch33.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import GammaRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(1, 10, size=(200, 2))
# توليد تكاليف موجبة ملتوية وفق توزيع غاما
mu = np.exp(1.0 + 0.3 * X[:, 0] + 0.2 * X[:, 1])
# شكل التوزيع k = 2
y = rng.gamma(shape=2.0, scale=mu / 2.0)

gamma_reg = GammaRegressor(alpha=0.1, max_iter=200).fit(X, y)

print(f"معاملات نموذج Gamma: {np.round(gamma_reg.coef_, 3)}")
print(f"الحد الثابت intercept_: {gamma_reg.intercept_:.3f}")
print(f"تنبؤ لعينة جديدة X=[2, 3]: {gamma_reg.predict([[2, 3]])[0]:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part04/ch33.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import TweedieRegressor
import numpy as np

rng = np.random.default_rng(42)
n = 300
X = rng.uniform(0, 5, size=(n, 2))
mu = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])

# محاكاة بيانات مطالبات تأمين (أصفار كثيرة + مبالغ موجبة)
has_claim = rng.binomial(n=1, p=np.clip(mu / 15.0, 0.05, 0.7))
claim_amount = rng.gamma(shape=2.0, scale=mu) * has_claim

# تدريب TweedieRegressor مع power=1.5
tweedie = TweedieRegressor(power=1.5, alpha=0.1, max_iter=300).fit(X, claim_amount)

print(f"نسبة العينات الصفرية في البيانات: {np.mean(claim_amount == 0):.1%}")
print(f"متوسط التكلفة المتنبأ بها: {tweedie.predict(X).mean():.3f}")
print(f"معاملات Tweedie: {np.round(tweedie.coef_, 3)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part04/ch33.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import d2_tweedie_score

rng = np.random.default_rng(42)
X = rng.normal(size=(300, 3))
y = rng.poisson(lam=np.exp(0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.0))

ols = LinearRegression().fit(X, y)
pois = PoissonRegressor(alpha=1e-3).fit(X, y)

pred_ols = ols.predict(X)
pred_pois = pois.predict(X)

print(f"عدد التنبؤات السالبة المستحيلة في OLS: {np.sum(pred_ols < 0)}")
print(f"عدد التنبؤات السالبة في Poisson:      {np.sum(pred_pois < 0)}")
print(f"D² Poisson score لـ OLS:     {d2_tweedie_score(y, np.clip(pred_ols, 1e-5, None), power=1):.4f}")
print(f"D² Poisson score لـ Poisson: {d2_tweedie_score(y, pred_pois, power=1):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part04/ch33.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import GammaRegressor
import numpy as np

# اختبار سلوك GammaRegressor عند وجود أصفار
X = np.array([[1], [2], [3]])
y_with_zero = np.array([0.0, 4.2, 5.1])

try:
    GammaRegressor().fit(X, y_with_zero)
except ValueError as err:
    print(f"خطأ ملتقط بنجاح: {err}")
    print("السبب الرياضي: دالة خسارة غاما تتضمن log(y) و 1/y، وكلاهما غير معرف عند الصفر!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-33-ch33.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.metrics import mean_poisson_deviance

rng = np.random.default_rng(42)
n_samples = 250
X = rng.uniform(0, 5, size=(n_samples, 2))

# معدل الحدوث الحقيقي يتبع دالة أسية (Poisson)
true_lambda = np.exp(0.5 + 0.6 * X[:, 0] - 0.4 * X[:, 1])
y = rng.poisson(lam=true_lambda)

ols = LinearRegression().fit(X, y)
poisson = PoissonRegressor(alpha=1e-3, max_iter=300).fit(X, y)

print(f"أدنى قيمة تنبأ بها OLS:     {ols.predict(X).min():.3f} (قد تكون سالبة!)")
print(f"أدنى قيمة تنبأ بها Poisson: {poisson.predict(X).min():.3f} (موجبة قطعاً)")
print(f"انحراف بواسون لـ OLS:       {mean_poisson_deviance(y, np.clip(ols.predict(X), 1e-6, None)):.4f}")
print(f"انحراف بواسون لـ Poisson:   {mean_poisson_deviance(y, poisson.predict(X)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-33-ch33.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import GammaRegressor
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(1, 10, size=(200, 2))
# توليد تكاليف موجبة ملتوية وفق توزيع غاما
mu = np.exp(1.0 + 0.3 * X[:, 0] + 0.2 * X[:, 1])
# شكل التوزيع k = 2
y = rng.gamma(shape=2.0, scale=mu / 2.0)

gamma_reg = GammaRegressor(alpha=0.1, max_iter=200).fit(X, y)

print(f"معاملات نموذج Gamma: {np.round(gamma_reg.coef_, 3)}")
print(f"الحد الثابت intercept_: {gamma_reg.intercept_:.3f}")
print(f"تنبؤ لعينة جديدة X=[2, 3]: {gamma_reg.predict([[2, 3]])[0]:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-33-ch33.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import TweedieRegressor
import numpy as np

rng = np.random.default_rng(42)
n = 300
X = rng.uniform(0, 5, size=(n, 2))
mu = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])

# محاكاة بيانات مطالبات تأمين (أصفار كثيرة + مبالغ موجبة)
has_claim = rng.binomial(n=1, p=np.clip(mu / 15.0, 0.05, 0.7))
claim_amount = rng.gamma(shape=2.0, scale=mu) * has_claim

# تدريب TweedieRegressor مع power=1.5
tweedie = TweedieRegressor(power=1.5, alpha=0.1, max_iter=300).fit(X, claim_amount)

print(f"نسبة العينات الصفرية في البيانات: {np.mean(claim_amount == 0):.1%}")
print(f"متوسط التكلفة المتنبأ بها: {tweedie.predict(X).mean():.3f}")
print(f"معاملات Tweedie: {np.round(tweedie.coef_, 3)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-33-ch33.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import d2_tweedie_score

rng = np.random.default_rng(42)
X = rng.normal(size=(300, 3))
y = rng.poisson(lam=np.exp(0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.0))

ols = LinearRegression().fit(X, y)
pois = PoissonRegressor(alpha=1e-3).fit(X, y)

pred_ols = ols.predict(X)
pred_pois = pois.predict(X)

print(f"عدد التنبؤات السالبة المستحيلة في OLS: {np.sum(pred_ols < 0)}")
print(f"عدد التنبؤات السالبة في Poisson:      {np.sum(pred_pois < 0)}")
print(f"D² Poisson score لـ OLS:     {d2_tweedie_score(y, np.clip(pred_ols, 1e-5, None), power=1):.4f}")
print(f"D² Poisson score لـ Poisson: {d2_tweedie_score(y, pred_pois, power=1):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-33-ch33.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import GammaRegressor
import numpy as np

# اختبار سلوك GammaRegressor عند وجود أصفار
X = np.array([[1], [2], [3]])
y_with_zero = np.array([0.0, 4.2, 5.1])

try:
    GammaRegressor().fit(X, y_with_zero)
except ValueError as err:
    print(f"خطأ ملتقط بنجاح: {err}")
    print("السبب الرياضي: دالة خسارة غاما تتضمن log(y) و 1/y، وكلاهما غير معرف عند الصفر!")
