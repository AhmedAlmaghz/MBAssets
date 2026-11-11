# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : اختبار التسريب عمليًّا
# المصدر  : src/chapters/ch39_interpretability.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).
