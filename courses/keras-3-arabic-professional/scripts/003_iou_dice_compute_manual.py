# ──────────────────────────────────────────────────────────────────────────
# الفصل 22: الكشف والتجزئة الدلالية
# القسم   : حساب IoU وDice يدويًّا
# المصدر  : src/chapters/ch22_detection_segmentation.md:173
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def iou_dice(y_true, y_pred, threshold=0.5):
    """حساب IoU وDice من قناعين ثنائييّن."""
    t = (y_true > 0.5).astype(bool)
    p = (y_pred >= threshold).astype(bool)
    inter = int(np.logical_and(t, p).sum())
    union = int(np.logical_or(t, p).sum())
    iou = inter / max(union, 1)
    dice = 2 * inter / max(t.sum() + p.sum(), 1)
    return iou, dice

# ─── 1. قناعان: صحيح ومتنبَّأ به
y_true = np.zeros((16, 16), dtype="float32")
y_true[4:10, 4:10] = 1.0                     # مربّع 6×6
y_pred = np.zeros((16, 16), dtype="float32")
y_pred[5:11, 5:11] = 1.0                     # مربّع مُزاح 6×6

iou, dice = iou_dice(y_true, y_pred)
print(f"IoU = {iou:.3f}, Dice = {dice:.3f}")
# النتيجة المتوقعة: تداخل 5×5 = 25، اتحاد 6×6 + 6×6 - 25 = 47
# ⇒ IoU ≈ 0.532، Dice ≈ 0.690.

# ─── 2. علاقة Dice و IoU
check = dice / (2 - dice)
print(f"التحقّق: Dice/(2-Dice) = {check:.3f} ≈ IoU")
