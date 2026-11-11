# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.10 معايرة الاحتمالات
# المصدر  : src/chapters/ch41_project_vision.md:221
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def reliability(probs, labels, n_bins=10):
    """متوسّط الاحتمال مقابل النسبة الفعلية في كلّ صندوق."""
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        rows.append((avg_p, acc, int(mask.sum())))
    return rows

# محاكاة: احتمالات وتسميات
probs = np.random.rand(500).astype("float32")
labels = (np.random.rand(500) < probs).astype("int32")
rows = reliability(probs, labels)
for avg_p, acc, count in rows[:4]:
    print(f"bin prob={avg_p:.2f} actual={acc:.2f} n={count}")
# النتيجة المتوقعة: في النموذج المعايَر، avg_p ≈ acc في كلّ صندوق.
