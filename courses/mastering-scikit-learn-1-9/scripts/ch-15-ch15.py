#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحويلات اللاخطية والتوزيعية
Mastering scikit-learn 1.9 — Chapter 15 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-15-ch15 (lesson 15)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التحويلات اللاخطية والتوزيعية
Mastering scikit-learn 1.9 — Chapter 15 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-15-ch15 (lesson 15)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import PowerTransformer

rng = np.random.default_rng(0)
X = rng.lognormal(3.0, 0.9, 1500).reshape(-1, 1)


def skew(a):
    a = np.asarray(a, dtype=float).ravel()
    return float(((a - a.mean()) ** 3).mean() / a.std() ** 3)


print(f"قبل: skew={skew(X):+.3f}  mean={X.mean():.1f}  median={np.median(X):.1f}  max={X.max():.0f}")

pt = PowerTransformer(method="yeo-johnson").fit(X)
Z = pt.transform(X).to_numpy()
print(f"بعد : skew={skew(Z):+.4f}  mean={Z.mean():+.3f}  std={Z.std():.3f}")
print(f"الأسّ المختار lambdas_ = {np.round(pt.lambdas_, 4)}   (قريب من الصفر = لوغاريتم)")

pt_bc = PowerTransformer(method="box-cox").fit(X[:700])
print(f"box-cox lambdas_       = {np.round(pt_bc.lambdas_, 4)}")

pt_ns = PowerTransformer(method="yeo-johnson", standardize=False).fit(X)
print(f"مع standardize=False   = skew {skew(pt_ns.transform(X).to_numpy()):+.4f}")

try:
    PowerTransformer(method="box-cox").fit(np.array([[0.0], [1.0], [2.0]]))
except Exception as exc:
    print("\nbox-cox مع أصفار ->", type(exc).__name__, ":", str(exc).splitlines()[0][:95])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import QuantileTransformer

qu = QuantileTransformer(n_quantiles=200, output_distribution="uniform", random_state=0).fit(X)
Q = qu.transform(X).to_numpy()
qn = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(X)
Qn = qn.transform(X).to_numpy()

print(f"uniform: min={Q.min():.4f} max={Q.max():.4f} n_quantiles_={qu.n_quantiles_}")
print(f"normal : mean={Qn.mean():+.4f} std={Qn.std():.4f}  skew={skew(Qn):+.4f}")
print(f"الترتيب محفوظ تماماً: {bool((np.argsort(X.ravel()) == np.argsort(Q.ravel())).all())}")
print(f"أقصى انحراف عن [0,1] : {float(np.abs(Q - Q.min()).max() - 1.0):+.2e}")

qs = QuantileTransformer(n_quantiles=50, output_distribution="normal", subsample=200,
                         random_state=0).fit(X)
print(f"subsample=200       : std={qs.transform(X).to_numpy().std():.4f}")
try:
    QuantileTransformer(output_distribution="gaussian")
except Exception as exc:
    print("قيمة غير صالحة ->", type(exc).__name__, ":", str(exc).splitlines()[0][:95])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import KBinsDiscretizer

print("توزيع العمود الخام:", np.round(np.percentile(X.ravel(), [0, 25, 50, 75, 100]), 1).tolist())
for strat in ["uniform", "quantile", "kmeans"]:
    b = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strat).fit(X)
    counts = np.bincount(b.transform(X).to_numpy().ravel().astype(int), minlength=4)
    print(f"  {strat:9s} counts={counts.tolist()}  edges={np.round(b.bin_edges_[0], 1).tolist()}")

full = KBinsDiscretizer(n_bins=4, strategy="quantile").fit(X).bin_edges_[0]
sub = KBinsDiscretizer(n_bins=4, strategy="quantile", subsample=200, random_state=0).fit(X).bin_edges_[0]
print("\nالحدود بكل الصفوف     :", np.round(full, 2).tolist())
print("الحدود بـ subsample=200:", np.round(sub, 2).tolist())

dense = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile").fit(X)
print("\nencode='onehot-dense'  ->", dense.transform(X).shape, list(dense.get_feature_names_out()))
print("encode='ordinal'        -> واحد لكلّ صفّ")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import SplineTransformer

xs = np.linspace(0.5, 18, 8).reshape(-1, 1)
sp = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
out = sp.fit_transform(xs)
print(f"الدخل {xs.shape}  ->  الخرج {out.shape}   n_features_out_={sp.n_features_out_}")
print("العُقد knots:", np.round(sp.bsplines_[0].t, 2).tolist())
print(f"degree=1, n_knots=4 -> {SplineTransformer(n_knots=4, degree=1).fit_transform(xs).shape}")
print(f"degree=3, n_knots=4 -> {SplineTransformer(n_knots=4, degree=3).fit_transform(xs).shape}")

with sklearn.config_context(transform_output="default"):
    print("sparse_output=True ->",
          type(SplineTransformer(n_knots=4, sparse_output=True).fit_transform(xs)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

A = np.array([[3.0, 4.0], [5.0, 6.0]])
cases = [
    dict(degree=2, include_bias=False),
    dict(degree=2, include_bias=False, interaction_only=True),
    dict(degree=3, include_bias=False),
    dict(degree=2, include_bias=True),
]
for kw in cases:
    p = PolynomialFeatures(**kw).fit(A)
    print(f"  {str(kw)[:58]:60s} -> {p.transform(A).shape[1]:2d} عمود: {list(p.get_feature_names_out(['a', 'b']))}")

print("\n  growth formula (d^2 + 3d) / 2:")
for d in (2, 5, 10, 20):
    print(f"    d={d:2d} -> {(d * d + 3 * d) // 2:3d} عمود")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch15.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import TransformedTargetRegressor

rng2 = np.random.default_rng(0)
Xc = rng2.uniform(-3, 3, (800, 1))
yc = Xc[:, 0] ** 3 + rng2.normal(0, 0.5, 800)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=0)

print(f"R² على هدف تكعيبي  y = x³   (انحراف الهدف = 0.05، أي شبه متماثل)")
print("  خطّي خام            :", round(LinearRegression().fit(Xtr, ytr).score(Xte, yte), 4))
print("  خطّي + Spline       :", round(Pipeline([
    ("sp", SplineTransformer(n_knots=6, degree=3)),
    ("m", LinearRegression())]).fit(Xtr, ytr).score(Xte, yte), 4))
print("  خطّي + Polynomial³  :", round(Pipeline([
    ("pf", PolynomialFeatures(degree=3, include_bias=False)),
    ("sc", StandardScaler()),
    ("m", LinearRegression())]).fit(Xtr, ytr).score(Xte, yte), 4))
print("  شجري خام            :", round(DecisionTreeRegressor(random_state=0).fit(Xtr, ytr).score(Xte, yte), 4))

X2 = rng2.normal(2.0, 1.0, (800, 1))
y2 = np.exp(X2[:, 0]) + rng2.normal(0, 0.3, 800)
a, b, ya, yb2 = train_test_split(X2, y2, test_size=0.25, random_state=0)
print(f"\nR² على هدف لوغاريتمي  y = eˣ  (انحراف الهدف = "
      f"{float(((y2 - y2.mean()) ** 3).mean() / y2.std() ** 3):.2f})")
print("  خطّي خام            :", round(LinearRegression().fit(a, ya).score(b, yb2), 4))
print("  + TransformedTarget :", round(TransformedTargetRegressor(
    regressor=LinearRegression(), func=np.log, inverse_func=np.exp
).fit(a, ya).score(b, yb2), 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import PowerTransformer

rng = np.random.default_rng(0)
X = rng.lognormal(3.0, 0.9, 1500).reshape(-1, 1)


def skew(a):
    a = np.asarray(a, dtype=float).ravel()
    return float(((a - a.mean()) ** 3).mean() / a.std() ** 3)


print(f"قبل: skew={skew(X):+.3f}  mean={X.mean():.1f}  median={np.median(X):.1f}  max={X.max():.0f}")

pt = PowerTransformer(method="yeo-johnson").fit(X)
Z = pt.transform(X).to_numpy()
print(f"بعد : skew={skew(Z):+.4f}  mean={Z.mean():+.3f}  std={Z.std():.3f}")
print(f"الأسّ المختار lambdas_ = {np.round(pt.lambdas_, 4)}   (قريب من الصفر = لوغاريتم)")

pt_bc = PowerTransformer(method="box-cox").fit(X[:700])
print(f"box-cox lambdas_       = {np.round(pt_bc.lambdas_, 4)}")

pt_ns = PowerTransformer(method="yeo-johnson", standardize=False).fit(X)
print(f"مع standardize=False   = skew {skew(pt_ns.transform(X).to_numpy()):+.4f}")

try:
    PowerTransformer(method="box-cox").fit(np.array([[0.0], [1.0], [2.0]]))
except Exception as exc:
    print("\nbox-cox مع أصفار ->", type(exc).__name__, ":", str(exc).splitlines()[0][:95])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import QuantileTransformer

qu = QuantileTransformer(n_quantiles=200, output_distribution="uniform", random_state=0).fit(X)
Q = qu.transform(X).to_numpy()
qn = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(X)
Qn = qn.transform(X).to_numpy()

print(f"uniform: min={Q.min():.4f} max={Q.max():.4f} n_quantiles_={qu.n_quantiles_}")
print(f"normal : mean={Qn.mean():+.4f} std={Qn.std():.4f}  skew={skew(Qn):+.4f}")
print(f"الترتيب محفوظ تماماً: {bool((np.argsort(X.ravel()) == np.argsort(Q.ravel())).all())}")
print(f"أقصى انحراف عن [0,1] : {float(np.abs(Q - Q.min()).max() - 1.0):+.2e}")

qs = QuantileTransformer(n_quantiles=50, output_distribution="normal", subsample=200,
                         random_state=0).fit(X)
print(f"subsample=200       : std={qs.transform(X).to_numpy().std():.4f}")
try:
    QuantileTransformer(output_distribution="gaussian")
except Exception as exc:
    print("قيمة غير صالحة ->", type(exc).__name__, ":", str(exc).splitlines()[0][:95])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import KBinsDiscretizer

print("توزيع العمود الخام:", np.round(np.percentile(X.ravel(), [0, 25, 50, 75, 100]), 1).tolist())
for strat in ["uniform", "quantile", "kmeans"]:
    b = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strat).fit(X)
    counts = np.bincount(b.transform(X).to_numpy().ravel().astype(int), minlength=4)
    print(f"  {strat:9s} counts={counts.tolist()}  edges={np.round(b.bin_edges_[0], 1).tolist()}")

full = KBinsDiscretizer(n_bins=4, strategy="quantile").fit(X).bin_edges_[0]
sub = KBinsDiscretizer(n_bins=4, strategy="quantile", subsample=200, random_state=0).fit(X).bin_edges_[0]
print("\nالحدود بكل الصفوف     :", np.round(full, 2).tolist())
print("الحدود بـ subsample=200:", np.round(sub, 2).tolist())

dense = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile").fit(X)
print("\nencode='onehot-dense'  ->", dense.transform(X).shape, list(dense.get_feature_names_out()))
print("encode='ordinal'        -> واحد لكلّ صفّ")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import SplineTransformer

xs = np.linspace(0.5, 18, 8).reshape(-1, 1)
sp = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
out = sp.fit_transform(xs)
print(f"الدخل {xs.shape}  ->  الخرج {out.shape}   n_features_out_={sp.n_features_out_}")
print("العُقد knots:", np.round(sp.bsplines_[0].t, 2).tolist())
print(f"degree=1, n_knots=4 -> {SplineTransformer(n_knots=4, degree=1).fit_transform(xs).shape}")
print(f"degree=3, n_knots=4 -> {SplineTransformer(n_knots=4, degree=3).fit_transform(xs).shape}")

with sklearn.config_context(transform_output="default"):
    print("sparse_output=True ->",
          type(SplineTransformer(n_knots=4, sparse_output=True).fit_transform(xs)).__name__)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

A = np.array([[3.0, 4.0], [5.0, 6.0]])
cases = [
    dict(degree=2, include_bias=False),
    dict(degree=2, include_bias=False, interaction_only=True),
    dict(degree=3, include_bias=False),
    dict(degree=2, include_bias=True),
]
for kw in cases:
    p = PolynomialFeatures(**kw).fit(A)
    print(f"  {str(kw)[:58]:60s} -> {p.transform(A).shape[1]:2d} عمود: {list(p.get_feature_names_out(['a', 'b']))}")

print("\n  growth formula (d^2 + 3d) / 2:")
for d in (2, 5, 10, 20):
    print(f"    d={d:2d} -> {(d * d + 3 * d) // 2:3d} عمود")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import TransformedTargetRegressor

rng2 = np.random.default_rng(0)
Xc = rng2.uniform(-3, 3, (800, 1))
yc = Xc[:, 0] ** 3 + rng2.normal(0, 0.5, 800)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=0)

print(f"R² على هدف تكعيبي  y = x³   (انحراف الهدف = 0.05، أي شبه متماثل)")
print("  خطّي خام            :", round(LinearRegression().fit(Xtr, ytr).score(Xte, yte), 4))
print("  خطّي + Spline       :", round(Pipeline([
    ("sp", SplineTransformer(n_knots=6, degree=3)),
    ("m", LinearRegression())]).fit(Xtr, ytr).score(Xte, yte), 4))
print("  خطّي + Polynomial³  :", round(Pipeline([
    ("pf", PolynomialFeatures(degree=3, include_bias=False)),
    ("sc", StandardScaler()),
    ("m", LinearRegression())]).fit(Xtr, ytr).score(Xte, yte), 4))
print("  شجري خام            :", round(DecisionTreeRegressor(random_state=0).fit(Xtr, ytr).score(Xte, yte), 4))

X2 = rng2.normal(2.0, 1.0, (800, 1))
y2 = np.exp(X2[:, 0]) + rng2.normal(0, 0.3, 800)
a, b, ya, yb2 = train_test_split(X2, y2, test_size=0.25, random_state=0)
print(f"\nR² على هدف لوغاريتمي  y = eˣ  (انحراف الهدف = "
      f"{float(((y2 - y2.mean()) ** 3).mean() / y2.std() ** 3):.2f})")
print("  خطّي خام            :", round(LinearRegression().fit(a, ya).score(b, yb2), 4))
print("  + TransformedTarget :", round(TransformedTargetRegressor(
    regressor=LinearRegression(), func=np.log, inverse_func=np.exp
).fit(a, ya).score(b, yb2), 4))
