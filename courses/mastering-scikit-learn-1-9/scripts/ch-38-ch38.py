#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — شجرة الانحدار وتقسيم خطأ مربّع
Mastering scikit-learn 1.9 — Chapter 38 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-38-ch38 (lesson 38)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-38-ch38.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — شجرة الانحدار وتقسيم خطأ مربّع
Mastering scikit-learn 1.9 — Chapter 38 — Trees

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-38-ch38 (lesson 38)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# توليد دالة غير خطية جيبية
rng = np.random.default_rng(42)
X = np.sort(rng.uniform(-3, 3, size=(80, 1)), axis=0)
y = np.sin(X).ravel() + rng.normal(scale=0.1, size=80)

# تدريب شجرتي انحدار بعمقين مختلفين
tree_shallow = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
tree_deep = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)

print(f"عدد الأوراق في الشجرة الضحلة (depth=2): {tree_shallow.get_n_leaves()}")
print(f"MSE التدريب للشجرة الضحلة: {mean_squared_error(y, tree_shallow.predict(X)):.4f}")
print(f"عدد الأوراق في الشجرة العميقة (depth=5): {tree_deep.get_n_leaves()}")
print(f"MSE التدريب للشجرة العميقة: {mean_squared_error(y, tree_deep.predict(X)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# بيانات ذات اتجاه صاعد واضح (Linear Trend)
X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

# اختبار عينة استقرائية جديدة كلياً X=10.0 (خارج نطاق التدريب [1, 5])
X_extrapolate = np.array([[10.0]])

print(f"القيمة الحقيقية المتوقعة للاتجاه عند X=10.0: 20.0")
print(f"تنبؤ النموذج الخطي (LinearRegression):     {linear.predict(X_extrapolate)[0]:.1f} (استقراء مثالي)")
print(f"تنبؤ شجرة الانحدار (DecisionTreeRegressor): {tree.predict(X_extrapolate)[0]:.1f} (تجمّد عند متوسط آخر ورقة!)")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1500], data.target[:1500], test_size=0.2, random_state=42
)

# مقارنة بين معيار الخطأ المربع ومعيار الخطأ المطلق المقاوم للشواذ
tree_mse = DecisionTreeRegressor(criterion="squared_error", max_depth=6, random_state=42).fit(X_tr, y_tr)
tree_mae = DecisionTreeRegressor(criterion="absolute_error", max_depth=6, random_state=42).fit(X_tr, y_tr)

pred_mse = tree_mse.predict(X_te)
pred_mae = tree_mae.predict(X_te)

print(f"معيار squared_error | MSE الاختبار: {mean_squared_error(y_te, pred_mse):.4f} | MAE: {mean_absolute_error(y_te, pred_mse):.4f}")
print(f"معيار absolute_error | MSE الاختبار: {mean_squared_error(y_te, pred_mae):.4f} | MAE: {mean_absolute_error(y_te, pred_mae):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# مثال مصغر لفهم الحسابات الداخلية
X_toy = np.array([[1], [2], [3], [4], [6], [7]])
y_toy = np.array([1.0, 1.2, 1.1, 5.0, 5.2, 4.9])

tree_toy = DecisionTreeRegressor(max_depth=1).fit(X_toy, y_toy)

# استخراج عتبة الانقسام
threshold = tree_toy.tree_.threshold[0]
feature = tree_toy.tree_.feature[0]
left_value = tree_toy.tree_.value[1][0][0]
right_value = tree_toy.tree_.value[2][0][0]

print(f"التقسيم الأمثل الذي وجدته الشجرة: X <= {threshold:.2f}")
print(f"قيمة التنبؤ للفرع الأيسر (متوسط العينات 1,2,3): {left_value:.3f} (الحقيقي: {y_toy[:3].mean():.3f})")
print(f"قيمة التنبؤ للفرع الأيمن (متوسط العينات 4,5,6): {right_value:.3f} (الحقيقي: {y_toy[3:].mean():.3f})")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

best_depth = 1
lowest_rmse = float("inf")

for depth in [2, 4, 6, 8, 10, 14]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_tr, y_tr)
    rmse_tr = root_mean_squared_error(y_tr, reg.predict(X_tr))
    rmse_te = root_mean_squared_error(y_te, reg.predict(X_te))
    if rmse_te < lowest_rmse:
        lowest_rmse = rmse_te
        best_depth = depth
    print(f"عمق {depth:2d} | RMSE تدريب: {rmse_tr:.3f} | RMSE اختبار: {rmse_te:.3f}")

print(f"\nالعمق الأمثل للاختبار هو: {best_depth} بأدنى خطأ: {lowest_rmse:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part05/ch38.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeRegressor
import numpy as np

X = np.array([[10], [20], [30], [40], [50]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

reg = DecisionTreeRegressor(max_leaf_nodes=3, random_state=42).fit(X, y)

# 1. التنبؤ التلقائي المباشر
preds_direct = reg.predict(X)

# 2. التنبؤ اليدوي عبر أرقام الأوراق
leaf_indices = reg.apply(X)
preds_manual = np.array([reg.tree_.value[leaf_idx][0][0] for leaf_idx in leaf_indices])

print("تنبؤات predict المباشرة:", preds_direct)
print("تنبؤات مطابقة أوراق tree_.value:", preds_manual)
print(f"هل التطابق تام عبر كل العينات؟ {np.allclose(preds_direct, preds_manual)}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# توليد دالة غير خطية جيبية
rng = np.random.default_rng(42)
X = np.sort(rng.uniform(-3, 3, size=(80, 1)), axis=0)
y = np.sin(X).ravel() + rng.normal(scale=0.1, size=80)

# تدريب شجرتي انحدار بعمقين مختلفين
tree_shallow = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
tree_deep = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)

print(f"عدد الأوراق في الشجرة الضحلة (depth=2): {tree_shallow.get_n_leaves()}")
print(f"MSE التدريب للشجرة الضحلة: {mean_squared_error(y, tree_shallow.predict(X)):.4f}")
print(f"عدد الأوراق في الشجرة العميقة (depth=5): {tree_deep.get_n_leaves()}")
print(f"MSE التدريب للشجرة العميقة: {mean_squared_error(y, tree_deep.predict(X)):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# بيانات ذات اتجاه صاعد واضح (Linear Trend)
X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

# اختبار عينة استقرائية جديدة كلياً X=10.0 (خارج نطاق التدريب [1, 5])
X_extrapolate = np.array([[10.0]])

print(f"القيمة الحقيقية المتوقعة للاتجاه عند X=10.0: 20.0")
print(f"تنبؤ النموذج الخطي (LinearRegression):     {linear.predict(X_extrapolate)[0]:.1f} (استقراء مثالي)")
print(f"تنبؤ شجرة الانحدار (DecisionTreeRegressor): {tree.predict(X_extrapolate)[0]:.1f} (تجمّد عند متوسط آخر ورقة!)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1500], data.target[:1500], test_size=0.2, random_state=42
)

# مقارنة بين معيار الخطأ المربع ومعيار الخطأ المطلق المقاوم للشواذ
tree_mse = DecisionTreeRegressor(criterion="squared_error", max_depth=6, random_state=42).fit(X_tr, y_tr)
tree_mae = DecisionTreeRegressor(criterion="absolute_error", max_depth=6, random_state=42).fit(X_tr, y_tr)

pred_mse = tree_mse.predict(X_te)
pred_mae = tree_mae.predict(X_te)

print(f"معيار squared_error | MSE الاختبار: {mean_squared_error(y_te, pred_mse):.4f} | MAE: {mean_absolute_error(y_te, pred_mse):.4f}")
print(f"معيار absolute_error | MSE الاختبار: {mean_squared_error(y_te, pred_mae):.4f} | MAE: {mean_absolute_error(y_te, pred_mae):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# مثال مصغر لفهم الحسابات الداخلية
X_toy = np.array([[1], [2], [3], [4], [6], [7]])
y_toy = np.array([1.0, 1.2, 1.1, 5.0, 5.2, 4.9])

tree_toy = DecisionTreeRegressor(max_depth=1).fit(X_toy, y_toy)

# استخراج عتبة الانقسام
threshold = tree_toy.tree_.threshold[0]
feature = tree_toy.tree_.feature[0]
left_value = tree_toy.tree_.value[1][0][0]
right_value = tree_toy.tree_.value[2][0][0]

print(f"التقسيم الأمثل الذي وجدته الشجرة: X <= {threshold:.2f}")
print(f"قيمة التنبؤ للفرع الأيسر (متوسط العينات 1,2,3): {left_value:.3f} (الحقيقي: {y_toy[:3].mean():.3f})")
print(f"قيمة التنبؤ للفرع الأيمن (متوسط العينات 4,5,6): {right_value:.3f} (الحقيقي: {y_toy[3:].mean():.3f})")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

data = fetch_california_housing()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data[:1200], data.target[:1200], test_size=0.3, random_state=42
)

best_depth = 1
lowest_rmse = float("inf")

for depth in [2, 4, 6, 8, 10, 14]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_tr, y_tr)
    rmse_tr = root_mean_squared_error(y_tr, reg.predict(X_tr))
    rmse_te = root_mean_squared_error(y_te, reg.predict(X_te))
    if rmse_te < lowest_rmse:
        lowest_rmse = rmse_te
        best_depth = depth
    print(f"عمق {depth:2d} | RMSE تدريب: {rmse_tr:.3f} | RMSE اختبار: {rmse_te:.3f}")

print(f"\nالعمق الأمثل للاختبار هو: {best_depth} بأدنى خطأ: {lowest_rmse:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-38-ch38.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeRegressor
import numpy as np

X = np.array([[10], [20], [30], [40], [50]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

reg = DecisionTreeRegressor(max_leaf_nodes=3, random_state=42).fit(X, y)

# 1. التنبؤ التلقائي المباشر
preds_direct = reg.predict(X)

# 2. التنبؤ اليدوي عبر أرقام الأوراق
leaf_indices = reg.apply(X)
preds_manual = np.array([reg.tree_.value[leaf_idx][0][0] for leaf_idx in leaf_indices])

print("تنبؤات predict المباشرة:", preds_direct)
print("تنبؤات مطابقة أوراق tree_.value:", preds_manual)
print(f"هل التطابق تام عبر كل العينات؟ {np.allclose(preds_direct, preds_manual)}")
