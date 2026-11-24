#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التقسيم المتقدم: Watershed وGrabCut والتجميع
OpenCV 5 and AI Computer Vision — Segment

Book slug : opencv-5-ai-vision
Chapter   : ch-11-ch11-segment (lesson 11)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-11-ch11-segment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التقسيم المتقدم: Watershed وGrabCut والتجميع
OpenCV 5 and AI Computer Vision — Segment

Book slug : opencv-5-ai-vision
Chapter   : ch-11-ch11-segment (lesson 11)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-11-ch11-segment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التقسيم المتقدم: Watershed وGrabCut والتجميع
OpenCV 5 and AI Computer Vision — Segment

Book slug : opencv-5-ai-vision
Chapter   : ch-11-ch11-segment (lesson 11)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch11_segment.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/board.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
_, marks = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
marks = marks.astype(np.uint8)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch11_segment.md
# ────────────────────────────────────────────────────────────────────────
# 1) بناء النقاط المراس: خلفية + مركز الأجسام
_, sure_bg = cv2.threshold(dt, 0.01 * dt.max(), 255, cv2.THRESH_BINARY_INV)
sure_bg = sure_bg.astype(np.uint8)
_, sure_fg = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)

unknown = cv2.subtract(sure_bg, sure_fg)            # المناطق غير المؤكدة
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1                                # خلفية = 1
markers[unknown == 255] = 0                          # غير معروف = 0

# 2) تنفيذ الخوارزمية
img2 = img.copy()
markers = cv2.watershed(img2, markers)
img2[markers == -1] = [0, 0, 255]                    # الحدود بالأحمر

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch11_segment.md
# ────────────────────────────────────────────────────────────────────────
img3 = img.copy()
mask = np.zeros(img3.shape[:2], np.uint8)

bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
rect = (50, 30, 300, 250)              # x, y, w, h — يشمل الجسم مع هامش

cv2.grabCut(img3, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
cut = img3 * mask2[:, :, np.newaxis]

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch11_segment.md
# ────────────────────────────────────────────────────────────────────────
pix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
pix = np.float32(pix)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pix, 3, None, criteria, 3,
                                cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
seg = centers[labels.flatten()].reshape(img.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/board.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
_, marks = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
marks = marks.astype(np.uint8)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) بناء النقاط المراس: خلفية + مركز الأجسام
_, sure_bg = cv2.threshold(dt, 0.01 * dt.max(), 255, cv2.THRESH_BINARY_INV)
sure_bg = sure_bg.astype(np.uint8)
_, sure_fg = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)

unknown = cv2.subtract(sure_bg, sure_fg)            # المناطق غير المؤكدة
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1                                # خلفية = 1
markers[unknown == 255] = 0                          # غير معروف = 0

# 2) تنفيذ الخوارزمية
img2 = img.copy()
markers = cv2.watershed(img2, markers)
img2[markers == -1] = [0, 0, 255]                    # الحدود بالأحمر

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
img3 = img.copy()
mask = np.zeros(img3.shape[:2], np.uint8)

bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
rect = (50, 30, 300, 250)              # x, y, w, h — يشمل الجسم مع هامش

cv2.grabCut(img3, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
cut = img3 * mask2[:, :, np.newaxis]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
pix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
pix = np.float32(pix)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pix, 3, None, criteria, 3,
                                cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
seg = centers[labels.flatten()].reshape(img.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/board.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
_, marks = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
marks = marks.astype(np.uint8)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) بناء النقاط المراس: خلفية + مركز الأجسام
_, sure_bg = cv2.threshold(dt, 0.01 * dt.max(), 255, cv2.THRESH_BINARY_INV)
sure_bg = sure_bg.astype(np.uint8)
_, sure_fg = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)

unknown = cv2.subtract(sure_bg, sure_fg)            # المناطق غير المؤكدة
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1                                # خلفية = 1
markers[unknown == 255] = 0                          # غير معروف = 0

# 2) تنفيذ الخوارزمية
img2 = img.copy()
markers = cv2.watershed(img2, markers)
img2[markers == -1] = [0, 0, 255]                    # الحدود بالأحمر

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
img3 = img.copy()
mask = np.zeros(img3.shape[:2], np.uint8)

bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
rect = (50, 30, 300, 250)              # x, y, w, h — يشمل الجسم مع هامش

cv2.grabCut(img3, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
cut = img3 * mask2[:, :, np.newaxis]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-11-ch11-segment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
pix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
pix = np.float32(pix)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pix, 3, None, criteria, 3,
                                cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
seg = centers[labels.flatten()].reshape(img.shape)
