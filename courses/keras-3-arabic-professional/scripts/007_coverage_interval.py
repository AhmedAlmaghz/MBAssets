# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : تحقّق تغطية الفاصل
# المصدر  : src/chapters/ch42_project_timeseries.md:191
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def coverage(y_true, lower, upper):
    """نسبة القيم داخل الفاصل."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

# محاكاة
y_val_true = np.array(keras.ops.array(val_pred + residuals))
cov = coverage(y_val_true, lower, upper)
print(f"coverage = {cov:.2%} (target = 90%)")
# النتيجة المتوقعة: تغطية قريبة من 90% إن كان الفاصل صحيحًا.
