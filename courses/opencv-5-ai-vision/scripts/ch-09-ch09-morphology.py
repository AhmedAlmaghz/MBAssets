#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — المورفولوجيا الرياضية: الأشكال والبنكسات
OpenCV 5 and AI Computer Vision — Morphology

Book slug : opencv-5-ai-vision
Chapter   : ch-09-ch09-morphology (lesson 9)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-09-ch09-morphology.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — المورفولوجيا الرياضية: الأشكال والبنكسات
OpenCV 5 and AI Computer Vision — Morphology

Book slug : opencv-5-ai-vision
Chapter   : ch-09-ch09-morphology (lesson 9)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-09-ch09-morphology.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — المورفولوجيا الرياضية: الأشكال والبنكسات
OpenCV 5 and AI Computer Vision — Morphology

Book slug : opencv-5-ai-vision
Chapter   : ch-09-ch09-morphology (lesson 9)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

for name, shape in [("RECT", cv2.MORPH_RECT),
                    ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (11, 11))
    print("%-8s shape=%s nonzero=%d" % (name, k.shape, int(np.count_nonzero(k))))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
scene = make_belt()  # دالة بناء المشهد من الفصل السابق
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded = cv2.erode(bw, k)
dilated = cv2.dilate(bw, k)

print("أبيض قبل:", int(np.count_nonzero(bw)))
print("بعد التآكل:", int(np.count_nonzero(eroded)))
print("بعد الإتماء:", int(np.count_nonzero(dilated)))
print("عدد المكوّنات قبل:", cv2.connectedComponents(bw, 8)[0] - 1)
print("عدد المكوّنات بعد التآكل:", cv2.connectedComponents(eroded, 8)[0] - 1)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
for size in (3, 9):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, k)
    print("نافذة %d: بكسلات الحدود=%d" % (size, np.count_nonzero(grad)))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
def count_holes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), 8)
    h, w = mask.shape
    return sum(1 for i in range(1, n)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < w
               and stats[i, 1] + stats[i, 3] < h
               and stats[i, 4] >= 4)

rng = np.random.default_rng(11)
noisy = bw.copy()
noisy[rng.integers(0, 480, 500), rng.integers(0, 640, 500)] = 255
whites = np.argwhere(bw > 0)
for y, x in whites[rng.choice(len(whites), 12, replace=False)]:
    cv2.circle(noisy, (int(x), int(y)), int(rng.integers(2, 5)), 0, -1)

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_open)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

for name, m in [("الخام", noisy), ("بعد الفتح", opened), ("بعد الغلق", closed)]:
    print("%-10s مكوّنات=%3d ثقوب=%2d أبيض=%d"
          # % (name, cv2.connectedComponents(m, 8)[0] - 1,
             count_holes(m), np.count_nonzero(m)))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
areas = sorted([int(stats[i, cv2.CC_STAT_AREA])
                for i in range(1, n)], reverse=True)
print("الخام: مكوّنات=%d" % (cv2.connectedComponents(noisy, 8)[0] - 1))
print("بعد الفتح: مكوّنات=%d" % (cv2.connectedComponents(opened, 8)[0] - 1))
print("بعد الغلق: مكوّنات=%d | ثقوب=%d" % (n - 1, count_holes(closed)))
print("المساحات النهائية:", areas)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
tophat = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, k)
blackhat = cv2.morphologyEx(uneven, cv2.MORPH_BLACKHAT, k)
_, direct = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, bh_bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("متوسط Top-Hat: %.2f | انحرافه: %.2f" % (float(tophat.mean()), float(tophat.std())))
print("متوسط Black-Hat: %.2f | انحرافه: %.2f" % (float(blackhat.mean()), float(blackhat.std())))
print("نسبة الحبر — عتبة عامة: %.1f%%" % (100 * np.count_nonzero(direct) / direct.size))
print("نسبة الحبر — Black-Hat: %.1f%%" % (100 * np.count_nonzero(bh_bw) / bh_bw.size))
print("تطابق IoU — عتبة عامة: %.1f%%" % iou(direct, truth))
print("تطابق IoU — Black-Hat: %.1f%%" % iou(bh_bw, truth))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
y, x = np.unravel_index(np.argmax(dt), dt.shape)
print("أقصى مسافة: %.2f" % float(dt.max()))
print("موقعها: (%d, %d)" % (x, y))
print("عند مركز الدائرة الصغيرة: %.2f" % float(dt[100, 90]))
print("عند مركز الدائرة الكبيرة: %.2f" % float(dt[295, 270]))
print("عدد بكسلات dt > 20: %d" % int(np.count_nonzero(dt > 20)))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
dia = np.zeros((200, 200), np.uint8)
cv2.fillPoly(dia, [np.array([[100, 20], [180, 100], [100, 180], [20, 100]], np.int32)], 255)
base = int(np.count_nonzero(dia))
print("مساحة المعيّن قبل:", base)
for name, shape in [("RECT", cv2.MORPH_RECT), ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (21, 21))
    op = cv2.morphologyEx(dia, cv2.MORPH_OPEN, k)
    area = int(np.count_nonzero(op))
    print("%-8s عناصر=%3d مساحة=%d محتفظ=%.1f%%"
          # % (name, np.count_nonzero(k), area, 100.0 * area / base))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

for name, shape in [("RECT", cv2.MORPH_RECT),
                    ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (11, 11))
    print("%-8s shape=%s nonzero=%d" % (name, k.shape, int(np.count_nonzero(k))))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
scene = make_belt()  # دالة بناء المشهد من الفصل السابق
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded = cv2.erode(bw, k)
dilated = cv2.dilate(bw, k)

print("أبيض قبل:", int(np.count_nonzero(bw)))
print("بعد التآكل:", int(np.count_nonzero(eroded)))
print("بعد الإتماء:", int(np.count_nonzero(dilated)))
print("عدد المكوّنات قبل:", cv2.connectedComponents(bw, 8)[0] - 1)
print("عدد المكوّنات بعد التآكل:", cv2.connectedComponents(eroded, 8)[0] - 1)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
for size in (3, 9):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, k)
    print("نافذة %d: بكسلات الحدود=%d" % (size, np.count_nonzero(grad)))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def count_holes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), 8)
    h, w = mask.shape
    return sum(1 for i in range(1, n)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < w
               and stats[i, 1] + stats[i, 3] < h
               and stats[i, 4] >= 4)

rng = np.random.default_rng(11)
noisy = bw.copy()
noisy[rng.integers(0, 480, 500), rng.integers(0, 640, 500)] = 255
whites = np.argwhere(bw > 0)
for y, x in whites[rng.choice(len(whites), 12, replace=False)]:
    cv2.circle(noisy, (int(x), int(y)), int(rng.integers(2, 5)), 0, -1)

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_open)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

for name, m in [("الخام", noisy), ("بعد الفتح", opened), ("بعد الغلق", closed)]:
    print("%-10s مكوّنات=%3d ثقوب=%2d أبيض=%d"
          # % (name, cv2.connectedComponents(m, 8)[0] - 1,
             count_holes(m), np.count_nonzero(m)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
areas = sorted([int(stats[i, cv2.CC_STAT_AREA])
                for i in range(1, n)], reverse=True)
print("الخام: مكوّنات=%d" % (cv2.connectedComponents(noisy, 8)[0] - 1))
print("بعد الفتح: مكوّنات=%d" % (cv2.connectedComponents(opened, 8)[0] - 1))
print("بعد الغلق: مكوّنات=%d | ثقوب=%d" % (n - 1, count_holes(closed)))
print("المساحات النهائية:", areas)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
tophat = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, k)
blackhat = cv2.morphologyEx(uneven, cv2.MORPH_BLACKHAT, k)
_, direct = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, bh_bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("متوسط Top-Hat: %.2f | انحرافه: %.2f" % (float(tophat.mean()), float(tophat.std())))
print("متوسط Black-Hat: %.2f | انحرافه: %.2f" % (float(blackhat.mean()), float(blackhat.std())))
print("نسبة الحبر — عتبة عامة: %.1f%%" % (100 * np.count_nonzero(direct) / direct.size))
print("نسبة الحبر — Black-Hat: %.1f%%" % (100 * np.count_nonzero(bh_bw) / bh_bw.size))
print("تطابق IoU — عتبة عامة: %.1f%%" % iou(direct, truth))
print("تطابق IoU — Black-Hat: %.1f%%" % iou(bh_bw, truth))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
y, x = np.unravel_index(np.argmax(dt), dt.shape)
print("أقصى مسافة: %.2f" % float(dt.max()))
print("موقعها: (%d, %d)" % (x, y))
print("عند مركز الدائرة الصغيرة: %.2f" % float(dt[100, 90]))
print("عند مركز الدائرة الكبيرة: %.2f" % float(dt[295, 270]))
print("عدد بكسلات dt > 20: %d" % int(np.count_nonzero(dt > 20)))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
dia = np.zeros((200, 200), np.uint8)
cv2.fillPoly(dia, [np.array([[100, 20], [180, 100], [100, 180], [20, 100]], np.int32)], 255)
base = int(np.count_nonzero(dia))
print("مساحة المعيّن قبل:", base)
for name, shape in [("RECT", cv2.MORPH_RECT), ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (21, 21))
    op = cv2.morphologyEx(dia, cv2.MORPH_OPEN, k)
    area = int(np.count_nonzero(op))
    print("%-8s عناصر=%3d مساحة=%d محتفظ=%.1f%%"
          # % (name, np.count_nonzero(k), area, 100.0 * area / base))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — المورفولوجيا الرياضية: الأشكال والبنكسات
OpenCV 5 and AI Computer Vision — Morphology

Book slug : opencv-5-ai-vision
Chapter   : ch-09-ch09-morphology (lesson 9)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

for name, shape in [("RECT", cv2.MORPH_RECT),
                    ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (11, 11))
    print("%-8s shape=%s nonzero=%d" % (name, k.shape, int(np.count_nonzero(k))))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
scene = make_belt()  # دالة بناء المشهد من الفصل السابق
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded = cv2.erode(bw, k)
dilated = cv2.dilate(bw, k)

print("أبيض قبل:", int(np.count_nonzero(bw)))
print("بعد التآكل:", int(np.count_nonzero(eroded)))
print("بعد الإتماء:", int(np.count_nonzero(dilated)))
print("عدد المكوّنات قبل:", cv2.connectedComponents(bw, 8)[0] - 1)
print("عدد المكوّنات بعد التآكل:", cv2.connectedComponents(eroded, 8)[0] - 1)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
for size in (3, 9):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, k)
    print("نافذة %d: بكسلات الحدود=%d" % (size, np.count_nonzero(grad)))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
def count_holes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), 8)
    h, w = mask.shape
    return sum(1 for i in range(1, n)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < w
               and stats[i, 1] + stats[i, 3] < h
               and stats[i, 4] >= 4)

rng = np.random.default_rng(11)
noisy = bw.copy()
noisy[rng.integers(0, 480, 500), rng.integers(0, 640, 500)] = 255
whites = np.argwhere(bw > 0)
for y, x in whites[rng.choice(len(whites), 12, replace=False)]:
    cv2.circle(noisy, (int(x), int(y)), int(rng.integers(2, 5)), 0, -1)

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_open)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

for name, m in [("الخام", noisy), ("بعد الفتح", opened), ("بعد الغلق", closed)]:
    print("%-10s مكوّنات=%3d ثقوب=%2d أبيض=%d"
          # % (name, cv2.connectedComponents(m, 8)[0] - 1,
             count_holes(m), np.count_nonzero(m)))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
areas = sorted([int(stats[i, cv2.CC_STAT_AREA])
                for i in range(1, n)], reverse=True)
print("الخام: مكوّنات=%d" % (cv2.connectedComponents(noisy, 8)[0] - 1))
print("بعد الفتح: مكوّنات=%d" % (cv2.connectedComponents(opened, 8)[0] - 1))
print("بعد الغلق: مكوّنات=%d | ثقوب=%d" % (n - 1, count_holes(closed)))
print("المساحات النهائية:", areas)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
tophat = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, k)
blackhat = cv2.morphologyEx(uneven, cv2.MORPH_BLACKHAT, k)
_, direct = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, bh_bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("متوسط Top-Hat: %.2f | انحرافه: %.2f" % (float(tophat.mean()), float(tophat.std())))
print("متوسط Black-Hat: %.2f | انحرافه: %.2f" % (float(blackhat.mean()), float(blackhat.std())))
print("نسبة الحبر — عتبة عامة: %.1f%%" % (100 * np.count_nonzero(direct) / direct.size))
print("نسبة الحبر — Black-Hat: %.1f%%" % (100 * np.count_nonzero(bh_bw) / bh_bw.size))
print("تطابق IoU — عتبة عامة: %.1f%%" % iou(direct, truth))
print("تطابق IoU — Black-Hat: %.1f%%" % iou(bh_bw, truth))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
y, x = np.unravel_index(np.argmax(dt), dt.shape)
print("أقصى مسافة: %.2f" % float(dt.max()))
print("موقعها: (%d, %d)" % (x, y))
print("عند مركز الدائرة الصغيرة: %.2f" % float(dt[100, 90]))
print("عند مركز الدائرة الكبيرة: %.2f" % float(dt[295, 270]))
print("عدد بكسلات dt > 20: %d" % int(np.count_nonzero(dt > 20)))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch09_morphology.md
# ────────────────────────────────────────────────────────────────────────
dia = np.zeros((200, 200), np.uint8)
cv2.fillPoly(dia, [np.array([[100, 20], [180, 100], [100, 180], [20, 100]], np.int32)], 255)
base = int(np.count_nonzero(dia))
print("مساحة المعيّن قبل:", base)
for name, shape in [("RECT", cv2.MORPH_RECT), ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (21, 21))
    op = cv2.morphologyEx(dia, cv2.MORPH_OPEN, k)
    area = int(np.count_nonzero(op))
    print("%-8s عناصر=%3d مساحة=%d محتفظ=%.1f%%"
          # % (name, np.count_nonzero(k), area, 100.0 * area / base))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

for name, shape in [("RECT", cv2.MORPH_RECT),
                    ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (11, 11))
    print("%-8s shape=%s nonzero=%d" % (name, k.shape, int(np.count_nonzero(k))))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
scene = make_belt()  # دالة بناء المشهد من الفصل السابق
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded = cv2.erode(bw, k)
dilated = cv2.dilate(bw, k)

print("أبيض قبل:", int(np.count_nonzero(bw)))
print("بعد التآكل:", int(np.count_nonzero(eroded)))
print("بعد الإتماء:", int(np.count_nonzero(dilated)))
print("عدد المكوّنات قبل:", cv2.connectedComponents(bw, 8)[0] - 1)
print("عدد المكوّنات بعد التآكل:", cv2.connectedComponents(eroded, 8)[0] - 1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
for size in (3, 9):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, k)
    print("نافذة %d: بكسلات الحدود=%d" % (size, np.count_nonzero(grad)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def count_holes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), 8)
    h, w = mask.shape
    return sum(1 for i in range(1, n)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < w
               and stats[i, 1] + stats[i, 3] < h
               and stats[i, 4] >= 4)

rng = np.random.default_rng(11)
noisy = bw.copy()
noisy[rng.integers(0, 480, 500), rng.integers(0, 640, 500)] = 255
whites = np.argwhere(bw > 0)
for y, x in whites[rng.choice(len(whites), 12, replace=False)]:
    cv2.circle(noisy, (int(x), int(y)), int(rng.integers(2, 5)), 0, -1)

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_open)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

for name, m in [("الخام", noisy), ("بعد الفتح", opened), ("بعد الغلق", closed)]:
    print("%-10s مكوّنات=%3d ثقوب=%2d أبيض=%d"
          # % (name, cv2.connectedComponents(m, 8)[0] - 1,
             count_holes(m), np.count_nonzero(m)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
areas = sorted([int(stats[i, cv2.CC_STAT_AREA])
                for i in range(1, n)], reverse=True)
print("الخام: مكوّنات=%d" % (cv2.connectedComponents(noisy, 8)[0] - 1))
print("بعد الفتح: مكوّنات=%d" % (cv2.connectedComponents(opened, 8)[0] - 1))
print("بعد الغلق: مكوّنات=%d | ثقوب=%d" % (n - 1, count_holes(closed)))
print("المساحات النهائية:", areas)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
tophat = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, k)
blackhat = cv2.morphologyEx(uneven, cv2.MORPH_BLACKHAT, k)
_, direct = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, bh_bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("متوسط Top-Hat: %.2f | انحرافه: %.2f" % (float(tophat.mean()), float(tophat.std())))
print("متوسط Black-Hat: %.2f | انحرافه: %.2f" % (float(blackhat.mean()), float(blackhat.std())))
print("نسبة الحبر — عتبة عامة: %.1f%%" % (100 * np.count_nonzero(direct) / direct.size))
print("نسبة الحبر — Black-Hat: %.1f%%" % (100 * np.count_nonzero(bh_bw) / bh_bw.size))
print("تطابق IoU — عتبة عامة: %.1f%%" % iou(direct, truth))
print("تطابق IoU — Black-Hat: %.1f%%" % iou(bh_bw, truth))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
y, x = np.unravel_index(np.argmax(dt), dt.shape)
print("أقصى مسافة: %.2f" % float(dt.max()))
print("موقعها: (%d, %d)" % (x, y))
print("عند مركز الدائرة الصغيرة: %.2f" % float(dt[100, 90]))
print("عند مركز الدائرة الكبيرة: %.2f" % float(dt[295, 270]))
print("عدد بكسلات dt > 20: %d" % int(np.count_nonzero(dt > 20)))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-09-ch09-morphology.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
dia = np.zeros((200, 200), np.uint8)
cv2.fillPoly(dia, [np.array([[100, 20], [180, 100], [100, 180], [20, 100]], np.int32)], 255)
base = int(np.count_nonzero(dia))
print("مساحة المعيّن قبل:", base)
for name, shape in [("RECT", cv2.MORPH_RECT), ("ELLIPSE", cv2.MORPH_ELLIPSE),
                    ("CROSS", cv2.MORPH_CROSS)]:
    k = cv2.getStructuringElement(shape, (21, 21))
    op = cv2.morphologyEx(dia, cv2.MORPH_OPEN, k)
    area = int(np.count_nonzero(op))
    print("%-8s عناصر=%3d مساحة=%d محتفظ=%.1f%%"
          # % (name, np.count_nonzero(k), area, 100.0 * area / base))
