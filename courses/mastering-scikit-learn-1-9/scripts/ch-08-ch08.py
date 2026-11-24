#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع تمهيدي شامل: أسعار المنازل
Mastering scikit-learn 1.9 — Chapter 08 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-08-ch08 (lesson 8)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-08-ch08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع تمهيدي شامل: أسعار المنازل
Mastering scikit-learn 1.9 — Chapter 08 — Foundations

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-08-ch08 (lesson 8)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"إجمالي عدد المقاطعات السكنية: {df.shape[0]:,} عينة")
print(f"عدد الأعمدة المستقلة (الميزات): {len(housing.feature_names)}")
print("أسماء الميزات:", housing.feature_names)
print("المتغير المستهدف (الهدف):", housing.target_names[0])

print("\nإحصاءات الدخل المتوسط (MedInc) وقيمة المنازل (MedHouseVal):")
summary = df[["MedInc", "MedHouseVal"]].describe().round(3)
print(summary.to_string())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"حجم مصفوفة التدريب: {X_train.shape[0]:,} صفاً و {X_train.shape[1]} ميزات")
print(f"حجم مصفوفة الاختبار المحجوبة: {X_test.shape[0]:,} صفاً")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# بناء خط الأنابيب الخطي
linear_pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

# حساب جذر متوسط مربع الخطأ (RMSE) عبر 5 طيات
cv_rmse = -cross_val_score(
    linear_pipe,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print(f"متوسط RMSE للنموذج الخطي: {cv_rmse.mean():.4f}")
print(f"الانحراف المعياري للخطأ عبر الطيّات: {cv_rmse.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate

hgb_model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    random_state=42
)

# قياس كل من RMSE ومعامل التحديد R²
cv_scores = cross_validate(
    hgb_model,
    X_train,
    y_train,
    cv=5,
    scoring=["neg_root_mean_squared_error", "r2"]
)

rmse_hgb = -cv_scores["test_neg_root_mean_squared_error"].mean()
r2_hgb = cv_scores["test_r2"].mean()

print(f"متوسط RMSE لنموذج HistGradientBoosting: {rmse_hgb:.4f}")
print(f"معامل التحديد R² المتوسط: {r2_hgb:.4f}")
print(f"نسبة التحسن في تقليص الخطأ مقارنة بالنموذج الخطي: {((0.728 - rmse_hgb) / 0.728) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# تدريب النموذج على كامل مصفوفة التدريب
hgb_model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار المستقلة
y_pred = hgb_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = hgb_model.score(X_test, y_test)

print("النتائج الرسمية لنموذج الإنتاج على بيانات الاختبار:")
print(f"  - جذر متوسط مربع الخطأ (RMSE) : {final_rmse:.4f}  (~{final_rmse * 100_000:.0f}$)")
print(f"  - متوسط الخطأ المطلق (MAE)    : {final_mae:.4f}  (~{final_mae * 100_000:.0f}$)")
print(f"  - معامل التحديد (R²)         : {final_r2:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# البواقي = القيمة الحقيقية - القيمة المتنبأ بها
residuals = y_test - y_pred

mean_bias = np.mean(residuals)
std_bias = np.std(residuals)
accurate_within_50k = np.mean(np.abs(residuals) < 0.50) * 100

print("تقرير تشخيص البواقي (Residual Analysis):")
print(f"  - متوسط البواقي (التحيّز العام)       : {mean_bias:+.4f}")
print(f"  - الانحراف المعياري للخطأ            : {std_bias:.4f}")
print(f"  - نسبة التنبؤات ذات الخطأ الأقل من 50,000$ : {accurate_within_50k:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch08.md
# ────────────────────────────────────────────────────────────────────────
import joblib
import tempfile
from pathlib import Path

# حفظ النموذج في مسار مؤقت
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "housing_model.joblib"
    joblib.dump(hgb_model, model_path)

    # إعادة تحميل النموذج والتنبؤ بعيّنة جديدة
    loaded_model = joblib.load(model_path)
    sample_pred = loaded_model.predict(X_test[:3])

print("تم حفظ واسترجاع النموذج بنجاح تام.")
print("تنبؤات العينات الثلاث الأولى:", sample_pred.round(3))
print("القيم الفعلية المقابلة        :", y_test[:3].round(3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"إجمالي عدد المقاطعات السكنية: {df.shape[0]:,} عينة")
print(f"عدد الأعمدة المستقلة (الميزات): {len(housing.feature_names)}")
print("أسماء الميزات:", housing.feature_names)
print("المتغير المستهدف (الهدف):", housing.target_names[0])

print("\nإحصاءات الدخل المتوسط (MedInc) وقيمة المنازل (MedHouseVal):")
summary = df[["MedInc", "MedHouseVal"]].describe().round(3)
print(summary.to_string())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"حجم مصفوفة التدريب: {X_train.shape[0]:,} صفاً و {X_train.shape[1]} ميزات")
print(f"حجم مصفوفة الاختبار المحجوبة: {X_test.shape[0]:,} صفاً")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# بناء خط الأنابيب الخطي
linear_pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

# حساب جذر متوسط مربع الخطأ (RMSE) عبر 5 طيات
cv_rmse = -cross_val_score(
    linear_pipe,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print(f"متوسط RMSE للنموذج الخطي: {cv_rmse.mean():.4f}")
print(f"الانحراف المعياري للخطأ عبر الطيّات: {cv_rmse.std():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate

hgb_model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    random_state=42
)

# قياس كل من RMSE ومعامل التحديد R²
cv_scores = cross_validate(
    hgb_model,
    X_train,
    y_train,
    cv=5,
    scoring=["neg_root_mean_squared_error", "r2"]
)

rmse_hgb = -cv_scores["test_neg_root_mean_squared_error"].mean()
r2_hgb = cv_scores["test_r2"].mean()

print(f"متوسط RMSE لنموذج HistGradientBoosting: {rmse_hgb:.4f}")
print(f"معامل التحديد R² المتوسط: {r2_hgb:.4f}")
print(f"نسبة التحسن في تقليص الخطأ مقارنة بالنموذج الخطي: {((0.728 - rmse_hgb) / 0.728) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# تدريب النموذج على كامل مصفوفة التدريب
hgb_model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار المستقلة
y_pred = hgb_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = hgb_model.score(X_test, y_test)

print("النتائج الرسمية لنموذج الإنتاج على بيانات الاختبار:")
print(f"  - جذر متوسط مربع الخطأ (RMSE) : {final_rmse:.4f}  (~{final_rmse * 100_000:.0f}$)")
print(f"  - متوسط الخطأ المطلق (MAE)    : {final_mae:.4f}  (~{final_mae * 100_000:.0f}$)")
print(f"  - معامل التحديد (R²)         : {final_r2:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# البواقي = القيمة الحقيقية - القيمة المتنبأ بها
residuals = y_test - y_pred

mean_bias = np.mean(residuals)
std_bias = np.std(residuals)
accurate_within_50k = np.mean(np.abs(residuals) < 0.50) * 100

print("تقرير تشخيص البواقي (Residual Analysis):")
print(f"  - متوسط البواقي (التحيّز العام)       : {mean_bias:+.4f}")
print(f"  - الانحراف المعياري للخطأ            : {std_bias:.4f}")
print(f"  - نسبة التنبؤات ذات الخطأ الأقل من 50,000$ : {accurate_within_50k:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-08-ch08.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import joblib
import tempfile
from pathlib import Path

# حفظ النموذج في مسار مؤقت
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "housing_model.joblib"
    joblib.dump(hgb_model, model_path)

    # إعادة تحميل النموذج والتنبؤ بعيّنة جديدة
    loaded_model = joblib.load(model_path)
    sample_pred = loaded_model.predict(X_test[:3])

print("تم حفظ واسترجاع النموذج بنجاح تام.")
print("تنبؤات العينات الثلاث الأولى:", sample_pred.round(3))
print("القيم الفعلية المقابلة        :", y_test[:3].round(3))
