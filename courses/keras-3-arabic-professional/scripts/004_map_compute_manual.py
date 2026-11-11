# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب mAP يدويًّا لمربّعات محيط
# المصدر  : src/chapters/ch22_detection_segmentation.md:206
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def box_iou(a, b):
    """IoU بين مربّعين (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)

def simple_map(preds, truths, iou_threshold=0.5):
    """mAP مبسّط: نسبة التنبّؤات الصحيحة عند عتبة IoU."""
    hits = 0
    for p in preds:
        if any(box_iou(p, t) >= iou_threshold for t in truths):
            hits += 1
    return hits / max(len(truths), 1)

# ─── 1. تنبّؤات وحقيقة أرضيّة
truths = [(10, 10, 40, 40), (60, 60, 90, 90)]
preds = [(12, 11, 42, 41), (58, 62, 88, 92), (100, 100, 130, 130)]  # الثالث خطأ
print(f"mAP@0.5 = {simple_map(preds, truths):.3f}")
# النتيجة المتوقعة: 0.5 — من التنبّؤين الأولين تطابق مع الحقيقة،
# والثالث لا يطابق شيئًا (من 2 حقيقة تطابق 1).
