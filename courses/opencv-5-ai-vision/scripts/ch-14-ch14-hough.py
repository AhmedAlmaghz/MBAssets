#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — تحويل هوف: الخطوط والدوائر والقياسات
OpenCV 5 and AI Computer Vision — Hough

Book slug : opencv-5-ai-vision
Chapter   : ch-14-ch14-hough (lesson 14)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-14-ch14-hough.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — تحويل هوف: الخطوط والدوائر والقياسات
OpenCV 5 and AI Computer Vision — Hough

Book slug : opencv-5-ai-vision
Chapter   : ch-14-ch14-hough (lesson 14)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch14_hough.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img  = cv2.imread("assets/samples/building.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 180)

lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=110)
# lines.shape = (N, 1, 2): N خط، وكل خط (r, θ)
print("عدد الخطوط المرشحة:", len(lines))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch14_hough.md
# ────────────────────────────────────────────────────────────────────────
for l in lines[:10]:
    r, theta = l[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * r, b * r
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch14_hough.md
# ────────────────────────────────────────────────────────────────────────
linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                         minLineLength=50, maxLineGap=10)

for l in linesP[:20]:
    x1, y1, x2, y2 = l[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    length = np.hypot(x2 - x1, y2 - y1)
    angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch14_hough.md
# ────────────────────────────────────────────────────────────────────────
img2   = cv2.imread("assets/samples/board.jpg")
gray2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2  = cv2.medianBlur(gray2, 5)

circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=1,
                           minDist=30,
                           param1=120,   # عتبات Canny العليا
                           param2=35,    # عتبة التصويت (كتلة)
                           minRadius=10, maxRadius=80)

if circles is not None:
    for c in circles[0][:10]:
        x, y, r = c.astype(int)
        cv2.circle(img2, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch14_hough.md
# ────────────────────────────────────────────────────────────────────────
# مرجع معروف: قياس «مسطرة» بمسافة 10 مم بين خطين
def pixels_per_mm(p1, p2, real_mm=10.0):
    """p1, p2: نقطتا مركزي خط المسطرة المكتشفان بي هوف"""
    px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
    return px / real_mm

# بعد الهوف: رتب المقاطع أفقياً واعثر على خطيين متوازيين قريبين
horiz = [l[0] for l in linesP
         if abs(l[0][3] - l[0][1]) < 3]            # شبه أفقي
horiz.sort(key=lambda l: l[1])

if len(horiz) >= 2:
    ppm = pixels_per_mm((horiz[0][0], horiz[0][1]),
                        (horiz[1][0], horiz[1][1]))
print("بكسل لكل ملّيمتر:", round(ppm, 2))
print("عرض الجسم بالملّيمتر: %0.2f" % (168 / ppm))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-14-ch14-hough.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img  = cv2.imread("assets/samples/building.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 180)

lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=110)
# lines.shape = (N, 1, 2): N خط، وكل خط (r, θ)
print("عدد الخطوط المرشحة:", len(lines))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-14-ch14-hough.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
for l in lines[:10]:
    r, theta = l[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * r, b * r
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-14-ch14-hough.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                         minLineLength=50, maxLineGap=10)

for l in linesP[:20]:
    x1, y1, x2, y2 = l[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    length = np.hypot(x2 - x1, y2 - y1)
    angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-14-ch14-hough.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
img2   = cv2.imread("assets/samples/board.jpg")
gray2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2  = cv2.medianBlur(gray2, 5)

circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=1,
                           minDist=30,
                           param1=120,   # عتبات Canny العليا
                           param2=35,    # عتبة التصويت (كتلة)
                           minRadius=10, maxRadius=80)

if circles is not None:
    for c in circles[0][:10]:
        x, y, r = c.astype(int)
        cv2.circle(img2, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-14-ch14-hough.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# مرجع معروف: قياس «مسطرة» بمسافة 10 مم بين خطين
def pixels_per_mm(p1, p2, real_mm=10.0):
    """p1, p2: نقطتا مركزي خط المسطرة المكتشفان بي هوف"""
    px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
    return px / real_mm

# بعد الهوف: رتب المقاطع أفقياً واعثر على خطيين متوازيين قريبين
horiz = [l[0] for l in linesP
         if abs(l[0][3] - l[0][1]) < 3]            # شبه أفقي
horiz.sort(key=lambda l: l[1])

if len(horiz) >= 2:
    ppm = pixels_per_mm((horiz[0][0], horiz[0][1]),
                        (horiz[1][0], horiz[1][1]))
print("بكسل لكل ملّيمتر:", round(ppm, 2))
print("عرض الجسم بالملّيمتر: %0.2f" % (168 / ppm))
