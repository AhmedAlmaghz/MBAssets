# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.9 الانحراف وإعادة التدريب
# المصدر  : src/chapters/ch42_project_timeseries.md:249
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(train_residuals, recent_residuals):
    """نسبة تغيّر متوسّط الخطأ — مؤشّر انحراف."""
    base = float(np.mean(np.abs(train_residuals)))
    recent = float(np.mean(np.abs(recent_residuals)))
    return (recent - base) / max(base, 1e-8)

# محاكاة
base_res = np.random.randn(100).astype("float32")
recent_res = np.random.randn(100).astype("float32") * 1.4
score = drift_score(base_res, recent_res)
print(f"drift = {score:+.2%}")
if score > 0.20:
    print("تنبيه: انحراف — أعد التدريب على البيانات الأحدث")
# النتيجة المتوقعة: درجة انحراف إيجابيّة تُطلق تنبيهًا.
