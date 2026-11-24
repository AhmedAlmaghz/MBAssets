#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تحويل الهدف وتحويل المدخلات
Mastering scikit-learn 1.9 — Chapter 87 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-87-ch87 (lesson 87)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-87-ch87.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تحويل الهدف وتحويل المدخلات
Mastering scikit-learn 1.9 — Chapter 87 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-87-ch87 (lesson 87)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             r2_score)

X, z = make_regression(n_samples=300, n_features=5, noise=25, random_state=42)
rng = np.random.default_rng(42)
y = np.exp(z / 40 + rng.normal(0.0, 0.30, size=z.size))
print(f"الهدف: أصغر قيمة {y.min():.3f} | وسيط {np.median(y):.2f} | أكبر {y.max():.2f}")

rows = {"خام": [], "محوَّل": []}
for seed in range(10):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=seed)
    models = {
        "خام": LinearRegression().fit(Xa, ya),
        "محوَّل": TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p,
            inverse_func=np.expm1).fit(Xa, ya),
    }
    for name, model in models.items():
        p = np.clip(model.predict(Xb), 0.0, None)
        rows[name].append([
            r2_score(yb, p), mean_absolute_error(yb, p),
            100 * mean_absolute_percentage_error(yb, p),
            r2_score(np.log1p(yb), np.log1p(p)),
        ])

for name, v in rows.items():
    v = np.array(v)
    print(f"{name}: R2={v[:,0].mean():+.3f} | MAE={v[:,1].mean():.2f} | "
          f"MAPE={v[:,2].mean():.0f}% | log-R2={v[:,3].mean():+.3f}")
print("R2 عبر الأقسام — خام  :", np.round(np.array(rows["خام"])[:, 0], 3))
print("R2 عبر الأقسام — محوَّل:", np.round(np.array(rows["محوَّل"])[:, 0], 3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42)
raw = LinearRegression().fit(Xa, ya)
ttr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                inverse_func=np.expm1).fit(Xa, ya)
print("نوع المحوّل الداخلي:", type(ttr.transformer_).__name__)
print("func:", ttr.transformer_.func, "| inverse_func:", ttr.transformer_.inverse_func)
head = ya[:4].reshape(-1, 1)
print("ما رآه الانحدار:", ttr.transformer_.transform(head).ravel())
print("log1p مباشرة    :", np.log1p(head).ravel())
print("round-trip      :", np.expm1(np.log1p(head)).ravel(), "| الأصل:",
      head.ravel())
print("معاملات خام   :", np.round(raw.coef_, 2))
print("معاملات محوَّل :", np.round(ttr.regressor_.coef_, 3))
print("score على مقياسه:", round(ttr.regressor_.score(Xa, np.log1p(ya)), 4),
      "| النموذج الخام:", round(raw.score(Xa, ya), 4))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.exp).fit(Xa, ya)
    print("تحذير:", str(caught[0].message)[:96])

try:
    TransformedTargetRegressor(regressor=LinearRegression(),
                               func=np.log1p).fit(Xa, ya)
except ValueError as exc:
    print("ValueError:", str(exc)[:118])

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def box_cox(v, lambdas=np.linspace(-2.0, 2.0, 201)):
    v = np.clip(np.asarray(v, dtype=float), 1e-6, None)
    best, best_ll = 0.0, -np.inf
    for lam in lambdas:
        t = np.log(v) if lam == 0 else (np.power(v, lam) - 1) / lam
        ll = float(np.mean(np.log(np.abs(lam) * np.power(v, lam - 1) + 1e-12)
                           + (lam - 1) * t - t))
        if ll > best_ll:
            best_ll, best = ll, lam
    return np.log(v) if best == 0 else (np.power(v, best) - 1) / best

t0 = time.perf_counter()
for _ in range(20):
    box_cox(y)
print("تكلفة box_cox لكل نداء (ملّي ثانية):",
      round((time.perf_counter() - t0) / 20 * 1000, 2))

grid = {"regressor__alpha": [0.1, 1.0, 10.0]}
for ci in (True, False):
    mdl = TransformedTargetRegressor(regressor=Ridge(), func=box_cox,
                                     inverse_func=lambda v: v, check_inverse=ci)
    g = GridSearchCV(mdl, grid, cv=3)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.fit(X, y)
    print(f"box_cox — check_inverse={ci}: {time.perf_counter() - t0:.3f} ثانية")

for ci in (True, False):
    mdl = TransformedTargetRegressor(regressor=Ridge(), func=np.log1p,
                                     inverse_func=np.expm1, check_inverse=ci)
    g = GridSearchCV(mdl, grid, cv=3)
    t0 = time.perf_counter(); g.fit(X, y)
    print(f"log1p — check_inverse={ci}: {time.perf_counter() - t0:.3f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
Xm, ym = make_regression(n_samples=250, n_features=4, n_targets=3,
                         noise=10, random_state=0)
Ym = np.exp(ym / 40)
Pa, Pb, Qa, Qb = train_test_split(Xm, Ym, test_size=0.25, random_state=42)
m3 = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                inverse_func=np.expm1).fit(Pa, Qa)
r3 = LinearRegression().fit(Pa, Qa)
print("شكل التنبؤ:", m3.predict(Pb).shape, "| معاملات الداخل:", m3.regressor_.coef_.shape)
print("مدى الهدف:", round(Ym.min(), 3), "إلى", round(Ym.max(), 1))
print("R2 لكل هدف — محوَّل:",
      np.round([r2_score(Qb[:, j], m3.predict(Pb)[:, j]) for j in range(3)], 3))
print("R2 لكل هدف — خام  :",
      np.round([r2_score(Qb[:, j], r3.predict(Pb)[:, j]) for j in range(3)], 3))
print("MAE — محوَّل:", round(mean_absolute_error(Qb, m3.predict(Pb)), 1),
      "| خام:", round(mean_absolute_error(Qb, r3.predict(Pb)), 1))
try:
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.mean,
                               inverse_func=np.mean).fit(Pa, Qa)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:100])
neg = Qa.copy(); neg[0, 0] = -0.5
try:
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.expm1).fit(Pa, neg)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[-1].strip()[:80])

# ────────────────────────────────────────────────────────────────────────
# [006] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
man = LinearRegression().fit(Xa, np.log1p(ya))
manual = np.expm1(man.predict(Xb))
print("تطابق الغلاف واليدوي:", bool(np.allclose(manual, ttr.predict(Xb))))
print("لو نُسي العكس — R2 على المقياس الخام:",
      round(r2_score(yb, man.predict(Xb)), 4))
print("الصحيح — R2 على المقياس الخام:", round(r2_score(yb, ttr.predict(Xb)), 4))
print("فقد في R2:", round(r2_score(yb, ttr.predict(Xb))
                          - r2_score(yb, man.predict(Xb)), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
model = LinearRegression().fit(X_train, np.log1p(y_train))

# في دفترTraining يعمل تماماً
report = np.expm1(model.predict(X_new))

# بعد شهر، في سكربت التشغيل، يعمل تماماً أيضاً — حتى أول مرّة يُعرض فيها رقم
report = model.predict(X_new)          # نسيت السطر، والقيم بين ٠٫١ و٤

# ────────────────────────────────────────────────────────────────────────
# [008] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

frame = pd.DataFrame({"visits": [3.0, 12.0, 40.0, 7.0, 25.0, 90.0],
                      "basket": [2.0, 5.0, 9.0, 4.0, 6.0, 3.0]})
target = np.array([12.0, 40.0, 150.0, 30.0, 90.0, 20.0])

def add_ratio(df):
    out = df.copy()
    out["per_visit"] = out["basket"] / out["visits"]
    return out

names_out = lambda _self, names: list(names) + ["per_visit"]
ratio_ft = FunctionTransformer(func=add_ratio, feature_names_out=names_out)
pipe = Pipeline([("log", FunctionTransformer(np.log1p,
                                             feature_names_out="one-to-one")),
                 ("ratio", ratio_ft),
                 ("sc", StandardScaler()),
                 ("reg", TransformedTargetRegressor(regressor=Ridge(alpha=1.0),
                                                    func=np.log1p,
                                                    inverse_func=np.expm1))]
                ).fit(frame, target)
print("أعمدة الأنبوب:", list(pipe[:-1].get_feature_names_out()))
print("أسماء المدخلات:", list(pipe.feature_names_in_))
print("ستة أمثلة للتنبؤ:", np.round(pipe.predict(frame), 1))
print("عدد بارامترات FunctionTransformer:",
      len(FunctionTransformer(np.sqrt).get_params(deep=False)))

# ────────────────────────────────────────────────────────────────────────
# [009] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
neg_frame = pd.DataFrame({"a": [1.0, 4.0, -9.0], "b": [2.0, -4.0, 16.0]})
bad = FunctionTransformer(np.sqrt, feature_names_out="one-to-one").fit_transform(neg_frame)
print(bad)
print("قيم مفقودة:", int(bad.isna().sum().sum()))
ok = FunctionTransformer(lambda v: np.sqrt(np.abs(v)),
                         feature_names_out="one-to-one").fit_transform(neg_frame)
print("بعد التصحيح:", ok.to_numpy().ravel())
print("validate=True نوع الناتج:",
      type(FunctionTransformer(np.log1p, validate=True).fit_transform(frame)).__name__)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    FunctionTransformer(np.log1p).fit_transform(np.asarray(frame))
    print("تحذير set_output:", str(caught[0].message)[:110])

# ────────────────────────────────────────────────────────────────────────
# [010] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def score_on(mdl, name):
    mdl.fit(Xa, ya)
    p = np.clip(mdl.predict(Xb), 0.0, None)
    print(f"{name}: R2={r2_score(yb, p):+.3f} | MAE={mean_absolute_error(yb, p):.2f} | "
          f"MAPE={100*mean_absolute_percentage_error(yb, p):.0f}%")

score_on(LinearRegression(), "بدون تحويل")
score_on(Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]),
         "تطبيع X فقط")
score_on(Pipeline([("pt", PowerTransformer(method="yeo-johnson")),
                   ("sc", StandardScaler()), ("reg", LinearRegression())]),
         "PowerTransformer على X")
score_on(Pipeline([("qt", QuantileTransformer(n_quantiles=100, random_state=0)),
                   ("reg", LinearRegression())]),
         "QuantileTransformer على X")
score_on(TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                    inverse_func=np.expm1),
         "تحويل الهدف فقط")

# ────────────────────────────────────────────────────────────────────────
# [011] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
X1, z1 = make_regression(n_samples=300, n_features=5, noise=25, random_state=42)
y1 = np.exp(z1 / 40 + np.random.default_rng(42).normal(0.0, 0.30, z1.size))
A1, B1, a1, b1 = train_test_split(X1, y1, test_size=0.25, random_state=42)

raw1 = LinearRegression().fit(A1, a1)
ttr1 = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                  inverse_func=np.expm1).fit(A1, a1)
for name, m in (("خام", raw1), ("محوَّل", ttr1)):
    p = np.clip(m.predict(B1), 0.0, None)
    print(f"{name}: MAE={mean_absolute_error(b1, p):.3f} | "
          f"MAPE={100 * mean_absolute_percentage_error(b1, p):.1f}% | "
          f"log-R2={r2_score(np.log1p(b1), np.log1p(p)):+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [012] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.exp).fit(A1, a1)
    print("length of warning:", len(caught), "|", str(caught[0].message)[:74])

gridx = {"regressor__fit_intercept": [True, False]}
for inv in (np.exp, np.expm1):
    m = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                   inverse_func=inv)
    t0 = time.perf_counter()
    GridSearchCV(m, gridx, cv=3).fit(A1, a1)
    print(f"inverse_func={inv.__name__}: {time.perf_counter() - t0:.3f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [013] content/part10/ch87.md
# ────────────────────────────────────────────────────────────────────────
frame2 = pd.DataFrame({"visits": [4.0, 9.0, 16.0, 25.0],
                        "basket": [3.0, -2.0, 7.0, 11.0]})
pipe2 = Pipeline([("root", FunctionTransformer(np.sqrt,
                                              feature_names_out="one-to-one")),
                  ("sc", StandardScaler()),
                  ("clf", Ridge(alpha=1.0))])
print("عدد الخطوات:", len(pipe2.steps), "| أسماء:", [n for n, _ in pipe2.steps])
transformed = pipe2[:-1].fit_transform(frame2)
print("قيم مفقودة بعد sqrt:", int(transformed.isna().sum().sum()))
try:
    pipe2.fit(frame2, [1.0, 2.0, 3.0, 4.0])
except ValueError as exc:
    print("Ridge رفض القيم المفقودة:", str(exc).splitlines()[0])
pipe3 = Pipeline([("root", FunctionTransformer(lambda x: np.sqrt(np.abs(x)),
                                              feature_names_out="one-to-one")),
                  ("sc", StandardScaler()),
                  ("clf", Ridge(alpha=1.0))])
pipe3.fit(frame2, [1.0, 2.0, 3.0, 4.0])
print("قيم مفقودة بعد التصحيح:",
      int(pipe3[:-1].transform(frame2).isna().sum().sum()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             r2_score)

X, z = make_regression(n_samples=300, n_features=5, noise=25, random_state=42)
rng = np.random.default_rng(42)
y = np.exp(z / 40 + rng.normal(0.0, 0.30, size=z.size))
print(f"الهدف: أصغر قيمة {y.min():.3f} | وسيط {np.median(y):.2f} | أكبر {y.max():.2f}")

rows = {"خام": [], "محوَّل": []}
for seed in range(10):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=seed)
    models = {
        "خام": LinearRegression().fit(Xa, ya),
        "محوَّل": TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p,
            inverse_func=np.expm1).fit(Xa, ya),
    }
    for name, model in models.items():
        p = np.clip(model.predict(Xb), 0.0, None)
        rows[name].append([
            r2_score(yb, p), mean_absolute_error(yb, p),
            100 * mean_absolute_percentage_error(yb, p),
            r2_score(np.log1p(yb), np.log1p(p)),
        ])

for name, v in rows.items():
    v = np.array(v)
    print(f"{name}: R2={v[:,0].mean():+.3f} | MAE={v[:,1].mean():.2f} | "
          f"MAPE={v[:,2].mean():.0f}% | log-R2={v[:,3].mean():+.3f}")
print("R2 عبر الأقسام — خام  :", np.round(np.array(rows["خام"])[:, 0], 3))
print("R2 عبر الأقسام — محوَّل:", np.round(np.array(rows["محوَّل"])[:, 0], 3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42)
raw = LinearRegression().fit(Xa, ya)
ttr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                inverse_func=np.expm1).fit(Xa, ya)
print("نوع المحوّل الداخلي:", type(ttr.transformer_).__name__)
print("func:", ttr.transformer_.func, "| inverse_func:", ttr.transformer_.inverse_func)
head = ya[:4].reshape(-1, 1)
print("ما رآه الانحدار:", ttr.transformer_.transform(head).ravel())
print("log1p مباشرة    :", np.log1p(head).ravel())
print("round-trip      :", np.expm1(np.log1p(head)).ravel(), "| الأصل:",
      head.ravel())
print("معاملات خام   :", np.round(raw.coef_, 2))
print("معاملات محوَّل :", np.round(ttr.regressor_.coef_, 3))
print("score على مقياسه:", round(ttr.regressor_.score(Xa, np.log1p(ya)), 4),
      "| النموذج الخام:", round(raw.score(Xa, ya), 4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.exp).fit(Xa, ya)
    print("تحذير:", str(caught[0].message)[:96])

try:
    TransformedTargetRegressor(regressor=LinearRegression(),
                               func=np.log1p).fit(Xa, ya)
except ValueError as exc:
    print("ValueError:", str(exc)[:118])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def box_cox(v, lambdas=np.linspace(-2.0, 2.0, 201)):
    v = np.clip(np.asarray(v, dtype=float), 1e-6, None)
    best, best_ll = 0.0, -np.inf
    for lam in lambdas:
        t = np.log(v) if lam == 0 else (np.power(v, lam) - 1) / lam
        ll = float(np.mean(np.log(np.abs(lam) * np.power(v, lam - 1) + 1e-12)
                           + (lam - 1) * t - t))
        if ll > best_ll:
            best_ll, best = ll, lam
    return np.log(v) if best == 0 else (np.power(v, best) - 1) / best

t0 = time.perf_counter()
for _ in range(20):
    box_cox(y)
print("تكلفة box_cox لكل نداء (ملّي ثانية):",
      round((time.perf_counter() - t0) / 20 * 1000, 2))

grid = {"regressor__alpha": [0.1, 1.0, 10.0]}
for ci in (True, False):
    mdl = TransformedTargetRegressor(regressor=Ridge(), func=box_cox,
                                     inverse_func=lambda v: v, check_inverse=ci)
    g = GridSearchCV(mdl, grid, cv=3)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.fit(X, y)
    print(f"box_cox — check_inverse={ci}: {time.perf_counter() - t0:.3f} ثانية")

for ci in (True, False):
    mdl = TransformedTargetRegressor(regressor=Ridge(), func=np.log1p,
                                     inverse_func=np.expm1, check_inverse=ci)
    g = GridSearchCV(mdl, grid, cv=3)
    t0 = time.perf_counter(); g.fit(X, y)
    print(f"log1p — check_inverse={ci}: {time.perf_counter() - t0:.3f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
Xm, ym = make_regression(n_samples=250, n_features=4, n_targets=3,
                         noise=10, random_state=0)
Ym = np.exp(ym / 40)
Pa, Pb, Qa, Qb = train_test_split(Xm, Ym, test_size=0.25, random_state=42)
m3 = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                inverse_func=np.expm1).fit(Pa, Qa)
r3 = LinearRegression().fit(Pa, Qa)
print("شكل التنبؤ:", m3.predict(Pb).shape, "| معاملات الداخل:", m3.regressor_.coef_.shape)
print("مدى الهدف:", round(Ym.min(), 3), "إلى", round(Ym.max(), 1))
print("R2 لكل هدف — محوَّل:",
      np.round([r2_score(Qb[:, j], m3.predict(Pb)[:, j]) for j in range(3)], 3))
print("R2 لكل هدف — خام  :",
      np.round([r2_score(Qb[:, j], r3.predict(Pb)[:, j]) for j in range(3)], 3))
print("MAE — محوَّل:", round(mean_absolute_error(Qb, m3.predict(Pb)), 1),
      "| خام:", round(mean_absolute_error(Qb, r3.predict(Pb)), 1))
try:
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.mean,
                               inverse_func=np.mean).fit(Pa, Qa)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:100])
neg = Qa.copy(); neg[0, 0] = -0.5
try:
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.expm1).fit(Pa, neg)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[-1].strip()[:80])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
man = LinearRegression().fit(Xa, np.log1p(ya))
manual = np.expm1(man.predict(Xb))
print("تطابق الغلاف واليدوي:", bool(np.allclose(manual, ttr.predict(Xb))))
print("لو نُسي العكس — R2 على المقياس الخام:",
      round(r2_score(yb, man.predict(Xb)), 4))
print("الصحيح — R2 على المقياس الخام:", round(r2_score(yb, ttr.predict(Xb)), 4))
print("فقد في R2:", round(r2_score(yb, ttr.predict(Xb))
                          - r2_score(yb, man.predict(Xb)), 4))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
model = LinearRegression().fit(X_train, np.log1p(y_train))

# في دفترTraining يعمل تماماً
report = np.expm1(model.predict(X_new))

# بعد شهر، في سكربت التشغيل، يعمل تماماً أيضاً — حتى أول مرّة يُعرض فيها رقم
report = model.predict(X_new)          # نسيت السطر، والقيم بين ٠٫١ و٤

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

frame = pd.DataFrame({"visits": [3.0, 12.0, 40.0, 7.0, 25.0, 90.0],
                      "basket": [2.0, 5.0, 9.0, 4.0, 6.0, 3.0]})
target = np.array([12.0, 40.0, 150.0, 30.0, 90.0, 20.0])

def add_ratio(df):
    out = df.copy()
    out["per_visit"] = out["basket"] / out["visits"]
    return out

names_out = lambda _self, names: list(names) + ["per_visit"]
ratio_ft = FunctionTransformer(func=add_ratio, feature_names_out=names_out)
pipe = Pipeline([("log", FunctionTransformer(np.log1p,
                                             feature_names_out="one-to-one")),
                 ("ratio", ratio_ft),
                 ("sc", StandardScaler()),
                 ("reg", TransformedTargetRegressor(regressor=Ridge(alpha=1.0),
                                                    func=np.log1p,
                                                    inverse_func=np.expm1))]
                ).fit(frame, target)
print("أعمدة الأنبوب:", list(pipe[:-1].get_feature_names_out()))
print("أسماء المدخلات:", list(pipe.feature_names_in_))
print("ستة أمثلة للتنبؤ:", np.round(pipe.predict(frame), 1))
print("عدد بارامترات FunctionTransformer:",
      len(FunctionTransformer(np.sqrt).get_params(deep=False)))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
neg_frame = pd.DataFrame({"a": [1.0, 4.0, -9.0], "b": [2.0, -4.0, 16.0]})
bad = FunctionTransformer(np.sqrt, feature_names_out="one-to-one").fit_transform(neg_frame)
print(bad)
print("قيم مفقودة:", int(bad.isna().sum().sum()))
ok = FunctionTransformer(lambda v: np.sqrt(np.abs(v)),
                         feature_names_out="one-to-one").fit_transform(neg_frame)
print("بعد التصحيح:", ok.to_numpy().ravel())
print("validate=True نوع الناتج:",
      type(FunctionTransformer(np.log1p, validate=True).fit_transform(frame)).__name__)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    FunctionTransformer(np.log1p).fit_transform(np.asarray(frame))
    print("تحذير set_output:", str(caught[0].message)[:110])

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def score_on(mdl, name):
    mdl.fit(Xa, ya)
    p = np.clip(mdl.predict(Xb), 0.0, None)
    print(f"{name}: R2={r2_score(yb, p):+.3f} | MAE={mean_absolute_error(yb, p):.2f} | "
          f"MAPE={100*mean_absolute_percentage_error(yb, p):.0f}%")

score_on(LinearRegression(), "بدون تحويل")
score_on(Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]),
         "تطبيع X فقط")
score_on(Pipeline([("pt", PowerTransformer(method="yeo-johnson")),
                   ("sc", StandardScaler()), ("reg", LinearRegression())]),
         "PowerTransformer على X")
score_on(Pipeline([("qt", QuantileTransformer(n_quantiles=100, random_state=0)),
                   ("reg", LinearRegression())]),
         "QuantileTransformer على X")
score_on(TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                    inverse_func=np.expm1),
         "تحويل الهدف فقط")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
X1, z1 = make_regression(n_samples=300, n_features=5, noise=25, random_state=42)
y1 = np.exp(z1 / 40 + np.random.default_rng(42).normal(0.0, 0.30, z1.size))
A1, B1, a1, b1 = train_test_split(X1, y1, test_size=0.25, random_state=42)

raw1 = LinearRegression().fit(A1, a1)
ttr1 = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                  inverse_func=np.expm1).fit(A1, a1)
for name, m in (("خام", raw1), ("محوَّل", ttr1)):
    p = np.clip(m.predict(B1), 0.0, None)
    print(f"{name}: MAE={mean_absolute_error(b1, p):.3f} | "
          f"MAPE={100 * mean_absolute_percentage_error(b1, p):.1f}% | "
          f"log-R2={r2_score(np.log1p(b1), np.log1p(p)):+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                               inverse_func=np.exp).fit(A1, a1)
    print("length of warning:", len(caught), "|", str(caught[0].message)[:74])

gridx = {"regressor__fit_intercept": [True, False]}
for inv in (np.exp, np.expm1):
    m = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p,
                                   inverse_func=inv)
    t0 = time.perf_counter()
    GridSearchCV(m, gridx, cv=3).fit(A1, a1)
    print(f"inverse_func={inv.__name__}: {time.perf_counter() - t0:.3f} ثانية")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-87-ch87.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
frame2 = pd.DataFrame({"visits": [4.0, 9.0, 16.0, 25.0],
                        "basket": [3.0, -2.0, 7.0, 11.0]})
pipe2 = Pipeline([("root", FunctionTransformer(np.sqrt,
                                              feature_names_out="one-to-one")),
                  ("sc", StandardScaler()),
                  ("clf", Ridge(alpha=1.0))])
print("عدد الخطوات:", len(pipe2.steps), "| أسماء:", [n for n, _ in pipe2.steps])
transformed = pipe2[:-1].fit_transform(frame2)
print("قيم مفقودة بعد sqrt:", int(transformed.isna().sum().sum()))
try:
    pipe2.fit(frame2, [1.0, 2.0, 3.0, 4.0])
except ValueError as exc:
    print("Ridge رفض القيم المفقودة:", str(exc).splitlines()[0])
pipe3 = Pipeline([("root", FunctionTransformer(lambda x: np.sqrt(np.abs(x)),
                                              feature_names_out="one-to-one")),
                  ("sc", StandardScaler()),
                  ("clf", Ridge(alpha=1.0))])
pipe3.fit(frame2, [1.0, 2.0, 3.0, 4.0])
print("قيم مفقودة بعد التصحيح:",
      int(pipe3[:-1].transform(frame2).isna().sum().sum()))
