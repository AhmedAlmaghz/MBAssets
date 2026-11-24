#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العتبات والتقسيم: من الإضاءة إلى القرار
OpenCV 5 and AI Computer Vision — Threshold

Book slug : opencv-5-ai-vision
Chapter   : ch-08-ch08-threshold (lesson 8)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-08-ch08-threshold.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العتبات والتقسيم: من الإضاءة إلى القرار
OpenCV 5 and AI Computer Vision — Threshold

Book slug : opencv-5-ai-vision
Chapter   : ch-08-ch08-threshold (lesson 8)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-08-ch08-threshold.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العتبات والتقسيم: من الإضاءة إلى القرار
OpenCV 5 and AI Computer Vision — Threshold

Book slug : opencv-5-ai-vision
Chapter   : ch-08-ch08-threshold (lesson 8)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
assert gray is not None, "فشلت قراءة الصورة: تحقق من المسار"

ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, bw_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("ret:", ret)
print("بيضاء BINARY:", int(np.count_nonzero(bw)))
print("بيضاء BINARY_INV:", int(np.count_nonzero(bw_inv)))
print("مجموع العدّين:", int(np.count_nonzero(bw) + np.count_nonzero(bw_inv)))
print("حجم الصورة:", gray.size)
print("البكسل (0, 0):", int(gray[0, 0]), "→", int(bw[0, 0]))
print("أقل قيمة:", int(gray.min()), "| أعلى قيمة:", int(gray.max()))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[127, 128]], np.uint8)
_, out = cv2.threshold(probe, 127, 255, cv2.THRESH_BINARY)
print(out.tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
bimodal = np.clip(rng.normal(60, 12, (200, 200)), 0, 255).astype(np.uint8)
shape = np.zeros_like(bimodal)
cv2.circle(shape, (60, 110), 42, 255, -1)
cv2.rectangle(shape, (110, 50), (190, 130), 255, -1)
idx = shape > 0
bimodal[idx] = np.clip(rng.normal(190, 14, int(idx.sum())), 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(bimodal, (5, 5), 1.0)

ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hist = np.bincount(blur.ravel(), minlength=256).astype(np.float64)
total = hist.sum()
w0 = np.cumsum(hist) / total
mu = np.cumsum(hist * np.arange(256)) / total
mu_t = mu[-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
sigma_b2[~np.isfinite(sigma_b2)] = 0
best = int(np.argmax(sigma_b2))

print("العتبة التي أعادها Otsu:", int(ret))
print("العتبة التي أعطت أكبر تباين بين الصنفين:", best)
print("وزن الصنف الداكن: %.4f" % w0[best])
print("متوسط الداكن: %.2f" % (mu[best] / w0[best]))
print("متوسط الفاتح: %.2f" % ((mu_t - mu[best]) / (1 - w0[best])))
print("التباين بين الصنفين: %.2f" % sigma_b2[best])
print("كفاءة الفصل eta: %.4f" % (sigma_b2[best] / float(blur.var())))
print("نسبة البكسلات الفاتحة: %.2f%%" % (100 * np.count_nonzero(mask) / mask.size))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
paper = cv2.GaussianBlur(cv2.imread("assets/samples/board.jpg", 0), (61, 61), 0)
paper = cv2.normalize(paper, None, 205, 235, cv2.NORM_MINMAX)
page, truth = paper.copy(), np.zeros_like(paper)
for i, text in enumerate(["NUR  QUALITY  CONTROL", "Batch 2026-04-17",
                          "Line 07  Shift B", "Inspector: Sara", "Status: PASS"]):
    org = (40, 55 + i * 55)
    cv2.putText(page, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 45, 2, cv2.LINE_AA)
    cv2.putText(truth, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
h, w = page.shape
ramp = np.linspace(0.45, 1.12, w, dtype=np.float32)[None, :]
uneven = np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

global_ret, global_bw = cv2.threshold(
    uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ad_mean = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)
ad_gauss = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

def iou(mask, truth):
    a, b = mask > 0, truth > 0
    return 100.0 * np.sum(a & b) / np.sum(a | b)

print("العتبة العامة:", int(global_ret))
print("تطابق IoU — عامة: %.1f%%" % iou(global_bw, truth))
print("تطابق IoU — تكيفية متوسط: %.1f%%" % iou(ad_mean, truth))
print("تطابق IoU — تكيفية غاوسي: %.1f%%" % iou(ad_gauss, truth))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
def make_belt(noise=False):
    base = cv2.imread("assets/samples/board.jpg")
    bg = cv2.convertScaleAbs(cv2.GaussianBlur(base, (41, 41), 0), alpha=0.18, beta=8)
    scene = bg.copy()
    for c, r in [((90, 100), 48), ((270, 295), 52)]:
        cv2.circle(scene, c, r, (0, 230, 255), -1)          # أصفر
    for p1, p2 in [((200, 50), (310, 140)), ((60, 240), (170, 340))]:
        cv2.rectangle(scene, p1, p2, (0, 60, 255), -1)      # أحمر برتقالي
    for pts in ([[420, 45], [490, 150], [350, 150]],
                [[430, 360], [510, 240], [350, 360]]):
        cv2.fillPoly(scene, [np.array(pts, np.int32)], (0, 180, 0))  # أخضر
    if noise:
        cv2.circle(scene, (600, 30), 2, (255, 255, 255), -1)
        cv2.circle(scene, (20, 380), 3, (255, 255, 255), -1)
    return cv2.GaussianBlur(scene, (3, 3), 0.8)

scene = make_belt()
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
red = cv2.bitwise_or(red1, red2)
green = cv2.inRange(hsv, np.array([40, 80, 60]), np.array([85, 255, 255]))

print("HSV عند مركز دائرة صفراء:", hsv[100, 90].tolist())
print("HSV عند مركز مستطيل أحمر:", hsv[95, 255].tolist())
print("HSV عند مركز مثلث أخضر:", hsv[115, 420].tolist())
print("بكسلات الأصفر:", int(np.count_nonzero(yellow)))
print("بكسلات الأحمر:", int(np.count_nonzero(red)))
print("بكسلات الأخضر:", int(np.count_nonzero(green)))

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
scene = make_belt(noise=True)
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
print("العتبة:", int(ret))
print("عدد المكوّنات الخام:", n - 1)

hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
kept, rejected = [], []
for i in range(1, n):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 500:
        rejected.append(area)
        continue
    kept.append(i)
    x, y, w, h, _ = [int(v) for v in stats[i]]
    cx, cy = [int(v) for v in centroids[i]]
    hue = int(hsv[cy, cx, 0])
    color = "أحمر" if hue <= 10 or hue >= 170 else ("أصفر" if 20 <= hue <= 35 else "أخضر")
    print("%d: x=%d y=%d w=%d h=%d area=%d centroid=(%d,%d) H=%d %s"
          # % (len(kept), x, y, w, h, area, cx, cy, hue, color))

print("المكوّنات المقبولة:", len(kept))
print("المساحات المرفوضة:", rejected)

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
for t in (80, 105, 140, 180):
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 500))
    print("t=%3d → %6d بكسل (%4.1f%%) | مكوّنات كبيرة=%d"
          # % (t, np.count_nonzero(bw), 100 * np.count_nonzero(bw) / bw.size, big))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
assert gray is not None, "فشلت قراءة الصورة: تحقق من المسار"

ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, bw_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("ret:", ret)
print("بيضاء BINARY:", int(np.count_nonzero(bw)))
print("بيضاء BINARY_INV:", int(np.count_nonzero(bw_inv)))
print("مجموع العدّين:", int(np.count_nonzero(bw) + np.count_nonzero(bw_inv)))
print("حجم الصورة:", gray.size)
print("البكسل (0, 0):", int(gray[0, 0]), "→", int(bw[0, 0]))
print("أقل قيمة:", int(gray.min()), "| أعلى قيمة:", int(gray.max()))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[127, 128]], np.uint8)
_, out = cv2.threshold(probe, 127, 255, cv2.THRESH_BINARY)
print(out.tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
bimodal = np.clip(rng.normal(60, 12, (200, 200)), 0, 255).astype(np.uint8)
shape = np.zeros_like(bimodal)
cv2.circle(shape, (60, 110), 42, 255, -1)
cv2.rectangle(shape, (110, 50), (190, 130), 255, -1)
idx = shape > 0
bimodal[idx] = np.clip(rng.normal(190, 14, int(idx.sum())), 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(bimodal, (5, 5), 1.0)

ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hist = np.bincount(blur.ravel(), minlength=256).astype(np.float64)
total = hist.sum()
w0 = np.cumsum(hist) / total
mu = np.cumsum(hist * np.arange(256)) / total
mu_t = mu[-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
sigma_b2[~np.isfinite(sigma_b2)] = 0
best = int(np.argmax(sigma_b2))

print("العتبة التي أعادها Otsu:", int(ret))
print("العتبة التي أعطت أكبر تباين بين الصنفين:", best)
print("وزن الصنف الداكن: %.4f" % w0[best])
print("متوسط الداكن: %.2f" % (mu[best] / w0[best]))
print("متوسط الفاتح: %.2f" % ((mu_t - mu[best]) / (1 - w0[best])))
print("التباين بين الصنفين: %.2f" % sigma_b2[best])
print("كفاءة الفصل eta: %.4f" % (sigma_b2[best] / float(blur.var())))
print("نسبة البكسلات الفاتحة: %.2f%%" % (100 * np.count_nonzero(mask) / mask.size))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
paper = cv2.GaussianBlur(cv2.imread("assets/samples/board.jpg", 0), (61, 61), 0)
paper = cv2.normalize(paper, None, 205, 235, cv2.NORM_MINMAX)
page, truth = paper.copy(), np.zeros_like(paper)
for i, text in enumerate(["NUR  QUALITY  CONTROL", "Batch 2026-04-17",
                          "Line 07  Shift B", "Inspector: Sara", "Status: PASS"]):
    org = (40, 55 + i * 55)
    cv2.putText(page, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 45, 2, cv2.LINE_AA)
    cv2.putText(truth, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
h, w = page.shape
ramp = np.linspace(0.45, 1.12, w, dtype=np.float32)[None, :]
uneven = np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

global_ret, global_bw = cv2.threshold(
    uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ad_mean = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)
ad_gauss = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

def iou(mask, truth):
    a, b = mask > 0, truth > 0
    return 100.0 * np.sum(a & b) / np.sum(a | b)

print("العتبة العامة:", int(global_ret))
print("تطابق IoU — عامة: %.1f%%" % iou(global_bw, truth))
print("تطابق IoU — تكيفية متوسط: %.1f%%" % iou(ad_mean, truth))
print("تطابق IoU — تكيفية غاوسي: %.1f%%" % iou(ad_gauss, truth))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def make_belt(noise=False):
    base = cv2.imread("assets/samples/board.jpg")
    bg = cv2.convertScaleAbs(cv2.GaussianBlur(base, (41, 41), 0), alpha=0.18, beta=8)
    scene = bg.copy()
    for c, r in [((90, 100), 48), ((270, 295), 52)]:
        cv2.circle(scene, c, r, (0, 230, 255), -1)          # أصفر
    for p1, p2 in [((200, 50), (310, 140)), ((60, 240), (170, 340))]:
        cv2.rectangle(scene, p1, p2, (0, 60, 255), -1)      # أحمر برتقالي
    for pts in ([[420, 45], [490, 150], [350, 150]],
                [[430, 360], [510, 240], [350, 360]]):
        cv2.fillPoly(scene, [np.array(pts, np.int32)], (0, 180, 0))  # أخضر
    if noise:
        cv2.circle(scene, (600, 30), 2, (255, 255, 255), -1)
        cv2.circle(scene, (20, 380), 3, (255, 255, 255), -1)
    return cv2.GaussianBlur(scene, (3, 3), 0.8)

scene = make_belt()
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
red = cv2.bitwise_or(red1, red2)
green = cv2.inRange(hsv, np.array([40, 80, 60]), np.array([85, 255, 255]))

print("HSV عند مركز دائرة صفراء:", hsv[100, 90].tolist())
print("HSV عند مركز مستطيل أحمر:", hsv[95, 255].tolist())
print("HSV عند مركز مثلث أخضر:", hsv[115, 420].tolist())
print("بكسلات الأصفر:", int(np.count_nonzero(yellow)))
print("بكسلات الأحمر:", int(np.count_nonzero(red)))
print("بكسلات الأخضر:", int(np.count_nonzero(green)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
scene = make_belt(noise=True)
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
print("العتبة:", int(ret))
print("عدد المكوّنات الخام:", n - 1)

hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
kept, rejected = [], []
for i in range(1, n):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 500:
        rejected.append(area)
        continue
    kept.append(i)
    x, y, w, h, _ = [int(v) for v in stats[i]]
    cx, cy = [int(v) for v in centroids[i]]
    hue = int(hsv[cy, cx, 0])
    color = "أحمر" if hue <= 10 or hue >= 170 else ("أصفر" if 20 <= hue <= 35 else "أخضر")
    print("%d: x=%d y=%d w=%d h=%d area=%d centroid=(%d,%d) H=%d %s"
          # % (len(kept), x, y, w, h, area, cx, cy, hue, color))

print("المكوّنات المقبولة:", len(kept))
print("المساحات المرفوضة:", rejected)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
for t in (80, 105, 140, 180):
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 500))
    print("t=%3d → %6d بكسل (%4.1f%%) | مكوّنات كبيرة=%d"
          # % (t, np.count_nonzero(bw), 100 * np.count_nonzero(bw) / bw.size, big))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العتبات والتقسيم: من الإضاءة إلى القرار
OpenCV 5 and AI Computer Vision — Threshold

Book slug : opencv-5-ai-vision
Chapter   : ch-08-ch08-threshold (lesson 8)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
assert gray is not None, "فشلت قراءة الصورة: تحقق من المسار"

ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, bw_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("ret:", ret)
print("بيضاء BINARY:", int(np.count_nonzero(bw)))
print("بيضاء BINARY_INV:", int(np.count_nonzero(bw_inv)))
print("مجموع العدّين:", int(np.count_nonzero(bw) + np.count_nonzero(bw_inv)))
print("حجم الصورة:", gray.size)
print("البكسل (0, 0):", int(gray[0, 0]), "→", int(bw[0, 0]))
print("أقل قيمة:", int(gray.min()), "| أعلى قيمة:", int(gray.max()))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[127, 128]], np.uint8)
_, out = cv2.threshold(probe, 127, 255, cv2.THRESH_BINARY)
print(out.tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
bimodal = np.clip(rng.normal(60, 12, (200, 200)), 0, 255).astype(np.uint8)
shape = np.zeros_like(bimodal)
cv2.circle(shape, (60, 110), 42, 255, -1)
cv2.rectangle(shape, (110, 50), (190, 130), 255, -1)
idx = shape > 0
bimodal[idx] = np.clip(rng.normal(190, 14, int(idx.sum())), 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(bimodal, (5, 5), 1.0)

ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hist = np.bincount(blur.ravel(), minlength=256).astype(np.float64)
total = hist.sum()
w0 = np.cumsum(hist) / total
mu = np.cumsum(hist * np.arange(256)) / total
mu_t = mu[-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
sigma_b2[~np.isfinite(sigma_b2)] = 0
best = int(np.argmax(sigma_b2))

print("العتبة التي أعادها Otsu:", int(ret))
print("العتبة التي أعطت أكبر تباين بين الصنفين:", best)
print("وزن الصنف الداكن: %.4f" % w0[best])
print("متوسط الداكن: %.2f" % (mu[best] / w0[best]))
print("متوسط الفاتح: %.2f" % ((mu_t - mu[best]) / (1 - w0[best])))
print("التباين بين الصنفين: %.2f" % sigma_b2[best])
print("كفاءة الفصل eta: %.4f" % (sigma_b2[best] / float(blur.var())))
print("نسبة البكسلات الفاتحة: %.2f%%" % (100 * np.count_nonzero(mask) / mask.size))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
paper = cv2.GaussianBlur(cv2.imread("assets/samples/board.jpg", 0), (61, 61), 0)
paper = cv2.normalize(paper, None, 205, 235, cv2.NORM_MINMAX)
page, truth = paper.copy(), np.zeros_like(paper)
for i, text in enumerate(["NUR  QUALITY  CONTROL", "Batch 2026-04-17",
                          "Line 07  Shift B", "Inspector: Sara", "Status: PASS"]):
    org = (40, 55 + i * 55)
    cv2.putText(page, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 45, 2, cv2.LINE_AA)
    cv2.putText(truth, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
h, w = page.shape
ramp = np.linspace(0.45, 1.12, w, dtype=np.float32)[None, :]
uneven = np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

global_ret, global_bw = cv2.threshold(
    uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ad_mean = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)
ad_gauss = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

def iou(mask, truth):
    a, b = mask > 0, truth > 0
    return 100.0 * np.sum(a & b) / np.sum(a | b)

print("العتبة العامة:", int(global_ret))
print("تطابق IoU — عامة: %.1f%%" % iou(global_bw, truth))
print("تطابق IoU — تكيفية متوسط: %.1f%%" % iou(ad_mean, truth))
print("تطابق IoU — تكيفية غاوسي: %.1f%%" % iou(ad_gauss, truth))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
def make_belt(noise=False):
    base = cv2.imread("assets/samples/board.jpg")
    bg = cv2.convertScaleAbs(cv2.GaussianBlur(base, (41, 41), 0), alpha=0.18, beta=8)
    scene = bg.copy()
    for c, r in [((90, 100), 48), ((270, 295), 52)]:
        cv2.circle(scene, c, r, (0, 230, 255), -1)          # أصفر
    for p1, p2 in [((200, 50), (310, 140)), ((60, 240), (170, 340))]:
        cv2.rectangle(scene, p1, p2, (0, 60, 255), -1)      # أحمر برتقالي
    for pts in ([[420, 45], [490, 150], [350, 150]],
                [[430, 360], [510, 240], [350, 360]]):
        cv2.fillPoly(scene, [np.array(pts, np.int32)], (0, 180, 0))  # أخضر
    if noise:
        cv2.circle(scene, (600, 30), 2, (255, 255, 255), -1)
        cv2.circle(scene, (20, 380), 3, (255, 255, 255), -1)
    return cv2.GaussianBlur(scene, (3, 3), 0.8)

scene = make_belt()
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
red = cv2.bitwise_or(red1, red2)
green = cv2.inRange(hsv, np.array([40, 80, 60]), np.array([85, 255, 255]))

print("HSV عند مركز دائرة صفراء:", hsv[100, 90].tolist())
print("HSV عند مركز مستطيل أحمر:", hsv[95, 255].tolist())
print("HSV عند مركز مثلث أخضر:", hsv[115, 420].tolist())
print("بكسلات الأصفر:", int(np.count_nonzero(yellow)))
print("بكسلات الأحمر:", int(np.count_nonzero(red)))
print("بكسلات الأخضر:", int(np.count_nonzero(green)))

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
scene = make_belt(noise=True)
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
print("العتبة:", int(ret))
print("عدد المكوّنات الخام:", n - 1)

hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
kept, rejected = [], []
for i in range(1, n):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 500:
        rejected.append(area)
        continue
    kept.append(i)
    x, y, w, h, _ = [int(v) for v in stats[i]]
    cx, cy = [int(v) for v in centroids[i]]
    hue = int(hsv[cy, cx, 0])
    color = "أحمر" if hue <= 10 or hue >= 170 else ("أصفر" if 20 <= hue <= 35 else "أخضر")
    print("%d: x=%d y=%d w=%d h=%d area=%d centroid=(%d,%d) H=%d %s"
          # % (len(kept), x, y, w, h, area, cx, cy, hue, color))

print("المكوّنات المقبولة:", len(kept))
print("المساحات المرفوضة:", rejected)

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch08_threshold.md
# ────────────────────────────────────────────────────────────────────────
gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
for t in (80, 105, 140, 180):
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 500))
    print("t=%3d → %6d بكسل (%4.1f%%) | مكوّنات كبيرة=%d"
          # % (t, np.count_nonzero(bw), 100 * np.count_nonzero(bw) / bw.size, big))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
assert gray is not None, "فشلت قراءة الصورة: تحقق من المسار"

ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, bw_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("ret:", ret)
print("بيضاء BINARY:", int(np.count_nonzero(bw)))
print("بيضاء BINARY_INV:", int(np.count_nonzero(bw_inv)))
print("مجموع العدّين:", int(np.count_nonzero(bw) + np.count_nonzero(bw_inv)))
print("حجم الصورة:", gray.size)
print("البكسل (0, 0):", int(gray[0, 0]), "→", int(bw[0, 0]))
print("أقل قيمة:", int(gray.min()), "| أعلى قيمة:", int(gray.max()))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[127, 128]], np.uint8)
_, out = cv2.threshold(probe, 127, 255, cv2.THRESH_BINARY)
print(out.tolist())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
bimodal = np.clip(rng.normal(60, 12, (200, 200)), 0, 255).astype(np.uint8)
shape = np.zeros_like(bimodal)
cv2.circle(shape, (60, 110), 42, 255, -1)
cv2.rectangle(shape, (110, 50), (190, 130), 255, -1)
idx = shape > 0
bimodal[idx] = np.clip(rng.normal(190, 14, int(idx.sum())), 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(bimodal, (5, 5), 1.0)

ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hist = np.bincount(blur.ravel(), minlength=256).astype(np.float64)
total = hist.sum()
w0 = np.cumsum(hist) / total
mu = np.cumsum(hist * np.arange(256)) / total
mu_t = mu[-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
sigma_b2[~np.isfinite(sigma_b2)] = 0
best = int(np.argmax(sigma_b2))

print("العتبة التي أعادها Otsu:", int(ret))
print("العتبة التي أعطت أكبر تباين بين الصنفين:", best)
print("وزن الصنف الداكن: %.4f" % w0[best])
print("متوسط الداكن: %.2f" % (mu[best] / w0[best]))
print("متوسط الفاتح: %.2f" % ((mu_t - mu[best]) / (1 - w0[best])))
print("التباين بين الصنفين: %.2f" % sigma_b2[best])
print("كفاءة الفصل eta: %.4f" % (sigma_b2[best] / float(blur.var())))
print("نسبة البكسلات الفاتحة: %.2f%%" % (100 * np.count_nonzero(mask) / mask.size))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
paper = cv2.GaussianBlur(cv2.imread("assets/samples/board.jpg", 0), (61, 61), 0)
paper = cv2.normalize(paper, None, 205, 235, cv2.NORM_MINMAX)
page, truth = paper.copy(), np.zeros_like(paper)
for i, text in enumerate(["NUR  QUALITY  CONTROL", "Batch 2026-04-17",
                          "Line 07  Shift B", "Inspector: Sara", "Status: PASS"]):
    org = (40, 55 + i * 55)
    cv2.putText(page, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 45, 2, cv2.LINE_AA)
    cv2.putText(truth, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
h, w = page.shape
ramp = np.linspace(0.45, 1.12, w, dtype=np.float32)[None, :]
uneven = np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

global_ret, global_bw = cv2.threshold(
    uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ad_mean = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)
ad_gauss = cv2.adaptiveThreshold(
    uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

def iou(mask, truth):
    a, b = mask > 0, truth > 0
    return 100.0 * np.sum(a & b) / np.sum(a | b)

print("العتبة العامة:", int(global_ret))
print("تطابق IoU — عامة: %.1f%%" % iou(global_bw, truth))
print("تطابق IoU — تكيفية متوسط: %.1f%%" % iou(ad_mean, truth))
print("تطابق IoU — تكيفية غاوسي: %.1f%%" % iou(ad_gauss, truth))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def make_belt(noise=False):
    base = cv2.imread("assets/samples/board.jpg")
    bg = cv2.convertScaleAbs(cv2.GaussianBlur(base, (41, 41), 0), alpha=0.18, beta=8)
    scene = bg.copy()
    for c, r in [((90, 100), 48), ((270, 295), 52)]:
        cv2.circle(scene, c, r, (0, 230, 255), -1)          # أصفر
    for p1, p2 in [((200, 50), (310, 140)), ((60, 240), (170, 340))]:
        cv2.rectangle(scene, p1, p2, (0, 60, 255), -1)      # أحمر برتقالي
    for pts in ([[420, 45], [490, 150], [350, 150]],
                [[430, 360], [510, 240], [350, 360]]):
        cv2.fillPoly(scene, [np.array(pts, np.int32)], (0, 180, 0))  # أخضر
    if noise:
        cv2.circle(scene, (600, 30), 2, (255, 255, 255), -1)
        cv2.circle(scene, (20, 380), 3, (255, 255, 255), -1)
    return cv2.GaussianBlur(scene, (3, 3), 0.8)

scene = make_belt()
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
red = cv2.bitwise_or(red1, red2)
green = cv2.inRange(hsv, np.array([40, 80, 60]), np.array([85, 255, 255]))

print("HSV عند مركز دائرة صفراء:", hsv[100, 90].tolist())
print("HSV عند مركز مستطيل أحمر:", hsv[95, 255].tolist())
print("HSV عند مركز مثلث أخضر:", hsv[115, 420].tolist())
print("بكسلات الأصفر:", int(np.count_nonzero(yellow)))
print("بكسلات الأحمر:", int(np.count_nonzero(red)))
print("بكسلات الأخضر:", int(np.count_nonzero(green)))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
scene = make_belt(noise=True)
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
print("العتبة:", int(ret))
print("عدد المكوّنات الخام:", n - 1)

hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
kept, rejected = [], []
for i in range(1, n):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 500:
        rejected.append(area)
        continue
    kept.append(i)
    x, y, w, h, _ = [int(v) for v in stats[i]]
    cx, cy = [int(v) for v in centroids[i]]
    hue = int(hsv[cy, cx, 0])
    color = "أحمر" if hue <= 10 or hue >= 170 else ("أصفر" if 20 <= hue <= 35 else "أخضر")
    print("%d: x=%d y=%d w=%d h=%d area=%d centroid=(%d,%d) H=%d %s"
          # % (len(kept), x, y, w, h, area, cx, cy, hue, color))

print("المكوّنات المقبولة:", len(kept))
print("المساحات المرفوضة:", rejected)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-08-ch08-threshold.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
gray = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)
for t in (80, 105, 140, 180):
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 500))
    print("t=%3d → %6d بكسل (%4.1f%%) | مكوّنات كبيرة=%d"
          # % (t, np.count_nonzero(bw), 100 * np.count_nonzero(bw) / bw.size, big))
