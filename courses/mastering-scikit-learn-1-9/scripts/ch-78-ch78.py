#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس الانحدار
Mastering scikit-learn 1.9 — Chapter 78 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-78-ch78 (lesson 78)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-78-ch78.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مقاييس الانحدار
Mastering scikit-learn 1.9 — Chapter 78 — Evaluation

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-78-ch78 (lesson 78)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

y_true = np.array([100., 120., 110., 130., 105., 125., 115., 122.])
err = np.array([2., -1., 3., -2., 1., -3., 2., -1.])   # أخطاء عادية صغيرة
y_pred = y_true + err
print(f"أخطاء عادية : MSE={mean_squared_error(y_true, y_pred):.3f} | "
      f"RMSE={root_mean_squared_error(y_true, y_pred):.3f} | MAE={mean_absolute_error(y_true, y_pred):.3f}")

err2 = err.copy()
err2[6] = 20.0                                            # خطأ كبير واحد فقط
y_pred2 = y_true + err2
print(f"خطأ كبير واحد: MSE={mean_squared_error(y_true, y_pred2):.3f} | "
      f"RMSE={root_mean_squared_error(y_true, y_pred2):.3f} | MAE={mean_absolute_error(y_true, y_pred2):.3f}")

r_mse = mean_squared_error(y_true, y_pred2) / mean_squared_error(y_true, y_pred)
r_mae = mean_absolute_error(y_true, y_pred2) / mean_absolute_error(y_true, y_pred)
print(f"نمو الخطأ: MSE ×{r_mse:.1f} ضد MAE ×{r_mae:.1f} — الفجوة هي بصمة التربيع")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (mean_absolute_error, median_absolute_error, max_error)

err = np.array([1., -1., 2., -2., 1., -1., 30., -1., 2., 1.])  # شاذ واحد بقيمة 30
y_true = np.full(10, 50.) + err
y_pred = np.full(10, 50.)                                       # تنبؤ مثالي عدا الشاذ

print(f"MAE            = {mean_absolute_error(y_true, y_pred):.3f} (تضخّم بفعل نقطة واحدة)")
print(f"median_AE      = {median_absolute_error(y_true, y_pred):.3f} (الواقع المعتاد)")
print(f"max_error      = {max_error(y_true, y_pred):.1f} (أقصى الأسوأ — للسلامة)")
print(f"نسبة MAE إلى median_AE: {mean_absolute_error(y_true, y_pred) / median_absolute_error(y_true, y_pred):.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

y_true = np.array([100.0, 50.0, 0.01])   # عينتان عاديتان وقيمة صغيرة جداً
y_pred = np.array([104.0, 52.0, 0.05])   # أخطاء مطلقة ضئيلة: 4 و 2 و 0.04

print("الأخطاء المطلقة :", np.abs(y_true - y_pred))
print("النسب الفردية   :", np.abs((y_true - y_pred) / y_true))
print(f"MAE  = {mean_absolute_error(y_true, y_pred):.4f} (يبدو هادئاً)")
print(f"MAPE = {mean_absolute_percentage_error(y_true, y_pred):.4f} أي 136%!")

def smape(y, p):
    return np.mean(2.0 * np.abs(y - p) / (np.abs(y) + np.abs(p)))
print(f"sMAPE (يدوياً) = {smape(y_true, y_pred):.4f} (أهدأ لكنه ليس حلاً جذرياً)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_log_error, root_mean_squared_log_error

y_true = np.array([1000.])
under  = np.array([500.])    # نقص بقيمة 500
over   = np.array([1500.])   # زيادة بقيمة 500 — المقدار المطلق نفسه

msle_u = mean_squared_log_error(y_true, under)
msle_o = mean_squared_log_error(y_true, over)
print(f"نقص 500  : MSLE = {msle_u:.4f} (RMSLE = {root_mean_squared_log_error(y_true, under):.4f})")
print(f"زيادة 500: MSLE = {msle_o:.4f} (RMSLE = {root_mean_squared_log_error(y_true, over):.4f})")
print(f"نسبة العقوبة (نقص/زيادة) = {msle_u / msle_o:.2f}x")
print("السبب: log(1+1000/500) أكبر من log(1+500/1000) — المسافة اللوغاريتمية غير متماثلة")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
X_tr = rng.uniform(0, 10, (300, 1)); y_tr = 3 * X_tr[:, 0] + rng.normal(0, 1, 300)
X_te = rng.uniform(15, 25, (100, 1)); y_te = 3 * X_te[:, 0] + rng.normal(0, 1, 100)

tree = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
p = tree.predict(X_te)                    # الشجرة لا تستطيع الاستقراء خارج مجالها
print(f"R² داخل مجال التدريب : {tree.score(X_tr, y_tr):.4f} (خادع بجماله)")
print(f"R² خارج المجال (15-25): {r2_score(y_te, p):.4f}")
print(f"متوسط الحقيقة {y_te.mean():.2f} مقابل متوسط تنبؤ الشجرة {p.mean():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss, d2_pinball_score

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:900], data.target[:900], test_size=0.3, random_state=42
)

preds = {}
for q in [0.1, 0.5, 0.9]:
    m = QuantileRegressor(quantile=q, alpha=0.01, solver="highs").fit(X_tr, y_tr)
    p = m.predict(X_te)
    preds[q] = p
    print(f"q={q}: pinball={mean_pinball_loss(y_te, p, alpha=q):.4f} | "
          f"d²_pinball={d2_pinball_score(y_te, p, alpha=q):.4f}")

coverage = np.mean((y_te >= preds[0.1]) & (y_te <= preds[0.9]))
print(f"تغطية الفاصل [q10, q90] = {coverage:.3f} (المستهدف 0.80)")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, d2_absolute_error_score)

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:900], data.target[:900], test_size=0.3, random_state=42
)

rows = []
for name, model in [("Linear", LinearRegression()),
                    ("Tree", DecisionTreeRegressor(random_state=42))]:
    p = model.fit(X_tr, y_tr).predict(X_te)
    rows.append((name,
        root_mean_squared_error(y_te, p), mean_absolute_error(y_te, p),
        mean_absolute_percentage_error(y_te, p), r2_score(y_te, p),
        d2_absolute_error_score(y_te, p)))

print(f"{'نموذج':8s} {'RMSE':>7s} {'MAE':>7s} {'MAPE':>7s} {'R2':>7s} {'d2abs':>7s}")
for n, rmse, mae, mape, r2, d2 in rows:
    print(f"{n:8s} {rmse:7.3f} {mae:7.3f} {mape:7.3f} {r2:7.3f} {d2:7.3f}")
print("الحكم اختلف: RMSE و R2 للنموذج الخطي، و MAE و MAPE و d2abs للشجرة — المقياس يقلب الترتيب")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part09/ch78.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_absolute_error

def pinball(y, p, q):
    e = y - p
    return np.mean(np.maximum(q * e, (q - 1.0) * e))

rng = np.random.default_rng(42)
y = rng.normal(50, 8, 200) + rng.normal(0, 2, 200)
p = y + rng.normal(0, 3, 200)

for q in [0.1, 0.5, 0.9]:
    manual, sk = pinball(y, p, q), mean_pinball_loss(y, p, alpha=q)
    print(f"q={q}: يدوي={manual:.6f} | sklearn={sk:.6f} | متطابقان: {np.isclose(manual, sk)}")
print(f"علاقة q=0.5: pinball={pinball(y, p, 0.5):.6f} = نصف MAE={mean_absolute_error(y, p) / 2:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

y_true = np.array([100., 120., 110., 130., 105., 125., 115., 122.])
err = np.array([2., -1., 3., -2., 1., -3., 2., -1.])   # أخطاء عادية صغيرة
y_pred = y_true + err
print(f"أخطاء عادية : MSE={mean_squared_error(y_true, y_pred):.3f} | "
      f"RMSE={root_mean_squared_error(y_true, y_pred):.3f} | MAE={mean_absolute_error(y_true, y_pred):.3f}")

err2 = err.copy()
err2[6] = 20.0                                            # خطأ كبير واحد فقط
y_pred2 = y_true + err2
print(f"خطأ كبير واحد: MSE={mean_squared_error(y_true, y_pred2):.3f} | "
      f"RMSE={root_mean_squared_error(y_true, y_pred2):.3f} | MAE={mean_absolute_error(y_true, y_pred2):.3f}")

r_mse = mean_squared_error(y_true, y_pred2) / mean_squared_error(y_true, y_pred)
r_mae = mean_absolute_error(y_true, y_pred2) / mean_absolute_error(y_true, y_pred)
print(f"نمو الخطأ: MSE ×{r_mse:.1f} ضد MAE ×{r_mae:.1f} — الفجوة هي بصمة التربيع")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (mean_absolute_error, median_absolute_error, max_error)

err = np.array([1., -1., 2., -2., 1., -1., 30., -1., 2., 1.])  # شاذ واحد بقيمة 30
y_true = np.full(10, 50.) + err
y_pred = np.full(10, 50.)                                       # تنبؤ مثالي عدا الشاذ

print(f"MAE            = {mean_absolute_error(y_true, y_pred):.3f} (تضخّم بفعل نقطة واحدة)")
print(f"median_AE      = {median_absolute_error(y_true, y_pred):.3f} (الواقع المعتاد)")
print(f"max_error      = {max_error(y_true, y_pred):.1f} (أقصى الأسوأ — للسلامة)")
print(f"نسبة MAE إلى median_AE: {mean_absolute_error(y_true, y_pred) / median_absolute_error(y_true, y_pred):.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

y_true = np.array([100.0, 50.0, 0.01])   # عينتان عاديتان وقيمة صغيرة جداً
y_pred = np.array([104.0, 52.0, 0.05])   # أخطاء مطلقة ضئيلة: 4 و 2 و 0.04

print("الأخطاء المطلقة :", np.abs(y_true - y_pred))
print("النسب الفردية   :", np.abs((y_true - y_pred) / y_true))
print(f"MAE  = {mean_absolute_error(y_true, y_pred):.4f} (يبدو هادئاً)")
print(f"MAPE = {mean_absolute_percentage_error(y_true, y_pred):.4f} أي 136%!")

def smape(y, p):
    return np.mean(2.0 * np.abs(y - p) / (np.abs(y) + np.abs(p)))
print(f"sMAPE (يدوياً) = {smape(y_true, y_pred):.4f} (أهدأ لكنه ليس حلاً جذرياً)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_squared_log_error, root_mean_squared_log_error

y_true = np.array([1000.])
under  = np.array([500.])    # نقص بقيمة 500
over   = np.array([1500.])   # زيادة بقيمة 500 — المقدار المطلق نفسه

msle_u = mean_squared_log_error(y_true, under)
msle_o = mean_squared_log_error(y_true, over)
print(f"نقص 500  : MSLE = {msle_u:.4f} (RMSLE = {root_mean_squared_log_error(y_true, under):.4f})")
print(f"زيادة 500: MSLE = {msle_o:.4f} (RMSLE = {root_mean_squared_log_error(y_true, over):.4f})")
print(f"نسبة العقوبة (نقص/زيادة) = {msle_u / msle_o:.2f}x")
print("السبب: log(1+1000/500) أكبر من log(1+500/1000) — المسافة اللوغاريتمية غير متماثلة")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
X_tr = rng.uniform(0, 10, (300, 1)); y_tr = 3 * X_tr[:, 0] + rng.normal(0, 1, 300)
X_te = rng.uniform(15, 25, (100, 1)); y_te = 3 * X_te[:, 0] + rng.normal(0, 1, 100)

tree = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
p = tree.predict(X_te)                    # الشجرة لا تستطيع الاستقراء خارج مجالها
print(f"R² داخل مجال التدريب : {tree.score(X_tr, y_tr):.4f} (خادع بجماله)")
print(f"R² خارج المجال (15-25): {r2_score(y_te, p):.4f}")
print(f"متوسط الحقيقة {y_te.mean():.2f} مقابل متوسط تنبؤ الشجرة {p.mean():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss, d2_pinball_score

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:900], data.target[:900], test_size=0.3, random_state=42
)

preds = {}
for q in [0.1, 0.5, 0.9]:
    m = QuantileRegressor(quantile=q, alpha=0.01, solver="highs").fit(X_tr, y_tr)
    p = m.predict(X_te)
    preds[q] = p
    print(f"q={q}: pinball={mean_pinball_loss(y_te, p, alpha=q):.4f} | "
          f"d²_pinball={d2_pinball_score(y_te, p, alpha=q):.4f}")

coverage = np.mean((y_te >= preds[0.1]) & (y_te <= preds[0.9]))
print(f"تغطية الفاصل [q10, q90] = {coverage:.3f} (المستهدف 0.80)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, d2_absolute_error_score)

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:900], data.target[:900], test_size=0.3, random_state=42
)

rows = []
for name, model in [("Linear", LinearRegression()),
                    ("Tree", DecisionTreeRegressor(random_state=42))]:
    p = model.fit(X_tr, y_tr).predict(X_te)
    rows.append((name,
        root_mean_squared_error(y_te, p), mean_absolute_error(y_te, p),
        mean_absolute_percentage_error(y_te, p), r2_score(y_te, p),
        d2_absolute_error_score(y_te, p)))

print(f"{'نموذج':8s} {'RMSE':>7s} {'MAE':>7s} {'MAPE':>7s} {'R2':>7s} {'d2abs':>7s}")
for n, rmse, mae, mape, r2, d2 in rows:
    print(f"{n:8s} {rmse:7.3f} {mae:7.3f} {mape:7.3f} {r2:7.3f} {d2:7.3f}")
print("الحكم اختلف: RMSE و R2 للنموذج الخطي، و MAE و MAPE و d2abs للشجرة — المقياس يقلب الترتيب")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-78-ch78.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_absolute_error

def pinball(y, p, q):
    e = y - p
    return np.mean(np.maximum(q * e, (q - 1.0) * e))

rng = np.random.default_rng(42)
y = rng.normal(50, 8, 200) + rng.normal(0, 2, 200)
p = y + rng.normal(0, 3, 200)

for q in [0.1, 0.5, 0.9]:
    manual, sk = pinball(y, p, q), mean_pinball_loss(y, p, alpha=q)
    print(f"q={q}: يدوي={manual:.6f} | sklearn={sk:.6f} | متطابقان: {np.isclose(manual, sk)}")
print(f"علاقة q=0.5: pinball={pinball(y, p, 0.5):.6f} = نصف MAE={mean_absolute_error(y, p) / 2:.6f}")
