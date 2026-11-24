#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الرياضيات تحت الغطاء
Mastering scikit-learn 1.9 — Chapter 06 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-06-ch06 (lesson 6)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-06-ch06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — الرياضيات تحت الغطاء
Mastering scikit-learn 1.9 — Chapter 06 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-06-ch06 (lesson 6)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=6, n_features=3, noise=5.0, random_state=0)
n, d = X.shape
print(f"X.shape = {X.shape}   أي {n} عيّنة و {d} ميزة")
print(f"y.shape = {y.shape}   متجه واحد لكل عيّنة")

w = np.zeros(d)
b = 0.5
print("\nX @ w + b =", np.round(X @ w + b, 4), " (كلّها b لأن w أصفار)")

w = np.array([1.5, -0.5, 2.0])
print("مع w =", w, " يصبح:", np.round(X @ w + b, 3))

# ضرب المصفوفات: (n,d) @ (d,) -> (n,)
print("\nشكل X @ w :", (X @ w).shape)
print("شكل X.T @ y:", (X.T @ y).shape, " وهو متجه معاملات")
print("شكل X.T @ X :", (X.T @ X).shape, " وهو مصفوفة (d,d)")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

X_array = np.asarray(X)
Xb = np.column_stack([X_array[:, 0], X_array[:, 1] * 50.0, X_array[:, 2]])
print("مدى كل عمود:", np.round(Xb.max(0) - Xb.min(0), 3))

q = Xb[0] + 1.0
raw_d = np.linalg.norm(Xb - q, axis=1)

sc = StandardScaler().fit(Xb)
scaled_d = np.linalg.norm(sc.transform(Xb) - sc.transform(q.reshape(1, -1)), axis=1)

print("\nأقرب جار بالمقياس الخام      :", int(np.argmin(raw_d)), " بمسافة", round(raw_d.min(), 4))
print("أقرب جار بعد التحجيم        :", int(np.argmin(scaled_d)), " بمسافة", round(scaled_d.min(), 4))
print("\nالنقطتان متطابقتان تماماً في الأبعاد الأربعة الأولى؟",
      np.allclose(Xb[0], q))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from scipy import stats

z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("الكثافة عند نقاط:", np.round(stats.norm.pdf(z), 4))
print("التوزيع التراكمي  :", np.round(stats.norm.cdf(z), 4))
print("المئين ٩٧٫٥      : %.3f" % stats.norm.ppf(0.975))
print("\nالقاعدة ٩٥٪ تعني: |z| < %.2f يغطّي ٩٥٪ من التوزيع" % stats.norm.ppf(0.975))

rng = np.random.default_rng(3)
right_skewed = rng.lognormal(0.0, 0.8, 4000)
print("\nتوزيع لوغاريتمي (مائل):")
print("  المتوسّط %.3f   الوسيط %.3f   الانحراف %.3f"
      # % (right_skewed.mean(), np.median(right_skewed), right_skewed.std()))
print("  نسبة القيم فوق المتوسّط:", round(float((right_skewed > right_skewed.mean()).mean()), 3))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes

Xd, yd = load_diabetes(return_X_y=True)
scaler = StandardScaler().fit(Xd)
Xd = np.column_stack([np.ones(len(Xd)), scaler.transform(Xd)])


def mse(w):
    r = yd - Xd @ w
    return float(r @ r / len(yd))


def grad(w):
    return -2.0 * (Xd.T @ (yd - Xd @ w)) / len(yd)


w_ols = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)

w = np.zeros(Xd.shape[1])
eta = 0.08
path = []
for step in range(400):
    w -= eta * grad(w)
    path.append(mse(w))

print("حلّ مغلق  MSE:", round(mse(w_ols), 4))
print("تدريج: خطوة 1  MSE:", round(path[0], 1))
print("تدريج: خطوة 50  MSE:", round(path[49], 4))
print("تدريج: خطوة 400 MSE:", round(path[-1], 4))
print("\nالمعاملات بالطريقتين (أول خمسة):")
print("  حلّ مغلق :", np.round(w_ols[:5], 3))
print("  تدريج    :", np.round(w[:5], 3))
print("\nفرق نسبي في الخانة العشرية:", f"{np.abs(w_ols - w).max() / np.abs(w_ols).max():.2e}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor

Xq, yq = make_regression(n_samples=400, n_features=4, noise=10.0, random_state=1)
yq[:8] += 600.0                       # ثماني قيم خارجة متعمّدة

ols = LinearRegression().fit(Xq, yq)
r2 = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(Xq, yq)

e_ols = yq - ols.predict(Xq)
e_q = yq - r2.predict(Xq)
print("معامل الانحدار الخطي     RMSE: %.2f   MAE: %.2f" % (np.sqrt((e_ols ** 2).mean()), np.abs(e_ols).mean()))
print("معامل الوسيط (L1)         RMSE: %.2f   MAE: %.2f" % (np.sqrt((e_q ** 2).mean()), np.abs(e_q).mean()))
print("\nالخطآن تحت ثماني نقاط خارجة فقط؟",
      np.round(np.abs(e_ols[:8]).sum(), 1), "ضد", round(float(np.abs(e_q[:8]).sum()), 1))
print("حجم الخطأ على بقية النقاط :",
      round(float(np.abs(e_ols[8:]).mean()), 2), "ضد", round(float(np.abs(e_q[8:]).mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression

Xs2 = np.asarray(StandardScaler().fit_transform(X))
m0 = LinearRegression().fit(Xs2, y)

print(f"على {Xs2.shape[1]} ميزات:")
print(f"{'f':>3} {'corr(X,y)':>10} {'coef':>10}")
for j in range(Xs2.shape[1]):
    print(f"{j:>3} {np.corrcoef(Xs2[:, j], y)[0, 1]:>10.4f} {m0.coef_[j]:>10.4f}")

rng = np.random.default_rng(0)
X_array = np.asarray(X)
X6 = np.column_stack([X_array, X_array[:, 2] * 0.98 + rng.normal(0, 0.25, len(X))])
X6s = np.asarray(StandardScaler().fit_transform(X6))
m1 = LinearRegression().fit(X6s, y)

print("\nبعد إضافة عمود رابع قريب من الثالث:")
print(f"{'f':>3} {'corr(X,y)':>10} {'coef':>10}")
for j in range(X6s.shape[1]):
    print(f"{j:>3} {np.corrcoef(X6s[:, j], y)[0, 1]:>10.4f} {m1.coef_[j]:>10.4f}")
print("\ncond(X'X) قبل/بعد:", round(float(np.linalg.cond(Xs2.T @ Xs2)), 2),
      "->", round(float(np.linalg.cond(X6s.T @ X6s)), 2))
print("معامل f2 قبل:", round(m0.coef_[2], 3),
      " بعد:", round(m1.coef_[2], 3), "+", round(m1.coef_[3], 3))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch06.md
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)

print(f"{'d':>6} {'أصغر مسافة':>12} {'متوسّط المسافة':>16} {'النسبة':>9}")
for d in (1, 3, 10, 100, 1000):
    P = rng.normal(size=(2000, d))
    q = rng.normal(size=d)
    dist = np.linalg.norm(P - q, axis=1)
    ratio = dist.min() / dist.mean()
    print(f"{d:>6} {dist.min():>12.4f} {dist.mean():>16.4f} {ratio:>9.4f}")
print("\nكلّما اقتربت النسبة من 1، فقدت المسافة قدرتها على التمييز.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=6, n_features=3, noise=5.0, random_state=0)
n, d = X.shape
print(f"X.shape = {X.shape}   أي {n} عيّنة و {d} ميزة")
print(f"y.shape = {y.shape}   متجه واحد لكل عيّنة")

w = np.zeros(d)
b = 0.5
print("\nX @ w + b =", np.round(X @ w + b, 4), " (كلّها b لأن w أصفار)")

w = np.array([1.5, -0.5, 2.0])
print("مع w =", w, " يصبح:", np.round(X @ w + b, 3))

# ضرب المصفوفات: (n,d) @ (d,) -> (n,)
print("\nشكل X @ w :", (X @ w).shape)
print("شكل X.T @ y:", (X.T @ y).shape, " وهو متجه معاملات")
print("شكل X.T @ X :", (X.T @ X).shape, " وهو مصفوفة (d,d)")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

X_array = np.asarray(X)
Xb = np.column_stack([X_array[:, 0], X_array[:, 1] * 50.0, X_array[:, 2]])
print("مدى كل عمود:", np.round(Xb.max(0) - Xb.min(0), 3))

q = Xb[0] + 1.0
raw_d = np.linalg.norm(Xb - q, axis=1)

sc = StandardScaler().fit(Xb)
scaled_d = np.linalg.norm(sc.transform(Xb) - sc.transform(q.reshape(1, -1)), axis=1)

print("\nأقرب جار بالمقياس الخام      :", int(np.argmin(raw_d)), " بمسافة", round(raw_d.min(), 4))
print("أقرب جار بعد التحجيم        :", int(np.argmin(scaled_d)), " بمسافة", round(scaled_d.min(), 4))
print("\nالنقطتان متطابقتان تماماً في الأبعاد الأربعة الأولى؟",
      np.allclose(Xb[0], q))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from scipy import stats

z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("الكثافة عند نقاط:", np.round(stats.norm.pdf(z), 4))
print("التوزيع التراكمي  :", np.round(stats.norm.cdf(z), 4))
print("المئين ٩٧٫٥      : %.3f" % stats.norm.ppf(0.975))
print("\nالقاعدة ٩٥٪ تعني: |z| < %.2f يغطّي ٩٥٪ من التوزيع" % stats.norm.ppf(0.975))

rng = np.random.default_rng(3)
right_skewed = rng.lognormal(0.0, 0.8, 4000)
print("\nتوزيع لوغاريتمي (مائل):")
print("  المتوسّط %.3f   الوسيط %.3f   الانحراف %.3f"
      # % (right_skewed.mean(), np.median(right_skewed), right_skewed.std()))
print("  نسبة القيم فوق المتوسّط:", round(float((right_skewed > right_skewed.mean()).mean()), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_diabetes

Xd, yd = load_diabetes(return_X_y=True)
scaler = StandardScaler().fit(Xd)
Xd = np.column_stack([np.ones(len(Xd)), scaler.transform(Xd)])


def mse(w):
    r = yd - Xd @ w
    return float(r @ r / len(yd))


def grad(w):
    return -2.0 * (Xd.T @ (yd - Xd @ w)) / len(yd)


w_ols = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)

w = np.zeros(Xd.shape[1])
eta = 0.08
path = []
for step in range(400):
    w -= eta * grad(w)
    path.append(mse(w))

print("حلّ مغلق  MSE:", round(mse(w_ols), 4))
print("تدريج: خطوة 1  MSE:", round(path[0], 1))
print("تدريج: خطوة 50  MSE:", round(path[49], 4))
print("تدريج: خطوة 400 MSE:", round(path[-1], 4))
print("\nالمعاملات بالطريقتين (أول خمسة):")
print("  حلّ مغلق :", np.round(w_ols[:5], 3))
print("  تدريج    :", np.round(w[:5], 3))
print("\nفرق نسبي في الخانة العشرية:", f"{np.abs(w_ols - w).max() / np.abs(w_ols).max():.2e}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor

Xq, yq = make_regression(n_samples=400, n_features=4, noise=10.0, random_state=1)
yq[:8] += 600.0                       # ثماني قيم خارجة متعمّدة

ols = LinearRegression().fit(Xq, yq)
r2 = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(Xq, yq)

e_ols = yq - ols.predict(Xq)
e_q = yq - r2.predict(Xq)
print("معامل الانحدار الخطي     RMSE: %.2f   MAE: %.2f" % (np.sqrt((e_ols ** 2).mean()), np.abs(e_ols).mean()))
print("معامل الوسيط (L1)         RMSE: %.2f   MAE: %.2f" % (np.sqrt((e_q ** 2).mean()), np.abs(e_q).mean()))
print("\nالخطآن تحت ثماني نقاط خارجة فقط؟",
      np.round(np.abs(e_ols[:8]).sum(), 1), "ضد", round(float(np.abs(e_q[:8]).sum()), 1))
print("حجم الخطأ على بقية النقاط :",
      round(float(np.abs(e_ols[8:]).mean()), 2), "ضد", round(float(np.abs(e_q[8:]).mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression

Xs2 = np.asarray(StandardScaler().fit_transform(X))
m0 = LinearRegression().fit(Xs2, y)

print(f"على {Xs2.shape[1]} ميزات:")
print(f"{'f':>3} {'corr(X,y)':>10} {'coef':>10}")
for j in range(Xs2.shape[1]):
    print(f"{j:>3} {np.corrcoef(Xs2[:, j], y)[0, 1]:>10.4f} {m0.coef_[j]:>10.4f}")

rng = np.random.default_rng(0)
X_array = np.asarray(X)
X6 = np.column_stack([X_array, X_array[:, 2] * 0.98 + rng.normal(0, 0.25, len(X))])
X6s = np.asarray(StandardScaler().fit_transform(X6))
m1 = LinearRegression().fit(X6s, y)

print("\nبعد إضافة عمود رابع قريب من الثالث:")
print(f"{'f':>3} {'corr(X,y)':>10} {'coef':>10}")
for j in range(X6s.shape[1]):
    print(f"{j:>3} {np.corrcoef(X6s[:, j], y)[0, 1]:>10.4f} {m1.coef_[j]:>10.4f}")
print("\ncond(X'X) قبل/بعد:", round(float(np.linalg.cond(Xs2.T @ Xs2)), 2),
      "->", round(float(np.linalg.cond(X6s.T @ X6s)), 2))
print("معامل f2 قبل:", round(m0.coef_[2], 3),
      " بعد:", round(m1.coef_[2], 3), "+", round(m1.coef_[3], 3))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-06-ch06.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)

print(f"{'d':>6} {'أصغر مسافة':>12} {'متوسّط المسافة':>16} {'النسبة':>9}")
for d in (1, 3, 10, 100, 1000):
    P = rng.normal(size=(2000, d))
    q = rng.normal(size=d)
    dist = np.linalg.norm(P - q, axis=1)
    ratio = dist.min() / dist.mean()
    print(f"{d:>6} {dist.min():>12.4f} {dist.mean():>16.4f} {ratio:>9.4f}")
print("\nكلّما اقتربت النسبة من 1، فقدت المسافة قدرتها على التمييز.")
