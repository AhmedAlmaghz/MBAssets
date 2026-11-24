#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الكائنات المتصلة والكونتورات: قراءة الأشكال
OpenCV 5 and AI Computer Vision — Contours

Book slug : opencv-5-ai-vision
Chapter   : ch-10-ch10-contours (lesson 10)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-10-ch10-contours.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الكائنات المتصلة والكونتورات: قراءة الأشكال
OpenCV 5 and AI Computer Vision — Contours

Book slug : opencv-5-ai-vision
Chapter   : ch-10-ch10-contours (lesson 10)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-10-ch10-contours.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الكائنات المتصلة والكونتورات: قراءة الأشكال
OpenCV 5 and AI Computer Vision — Contours

Book slug : opencv-5-ai-vision
Chapter   : ch-10-ch10-contours (lesson 10)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

scene = make_belt()
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("عدد الكونتورات:", len(contours))
print("شكل القناع:", bw.shape, "| النوع:", bw.dtype)
print("مجموع النقاط:", sum(len(c) for c in contours))
print("مساحة أكبرها: %.1f" % cv2.contourArea(max(contours, key=cv2.contourArea)))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
none, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
simple, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_none = sum(len(c) for c in none)
n_simple = sum(len(c) for c in simple)
print("NONE:", n_none, "| SIMPLE:", n_simple)
print("التوفير: %.1f%%" % (100 * (1 - n_simple / n_none)))
print("أطول كونتور NONE:", max(len(c) for c in none))
print("أطول كونتور SIMPLE:", max(len(c) for c in simple))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
m = np.zeros((240, 360), np.uint8)
cv2.rectangle(m, (30, 30), (200, 210), 255, -1)
cv2.rectangle(m, (60, 60), (170, 180), 0, -1)
cv2.circle(m, (115, 120), 30, 255, -1)
cv2.circle(m, (280, 120), 60, 255, -1)
cv2.circle(m, (280, 120), 25, 0, -1)

for name, mode in [("EXTERNAL", cv2.RETR_EXTERNAL),
                   ("LIST", cv2.RETR_LIST),
                   ("CCOMP", cv2.RETR_CCOMP),
                   ("TREE", cv2.RETR_TREE)]:
    cs, h = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    hh = h.reshape(-1, 4)
    depths = []
    for i in range(len(cs)):
        d, p = 0, int(hh[i, 3])
        while p != -1:
            d += 1
            p = int(hh[p, 3])
        depths.append(d)
    print("%-8s عدد=%d أقصى عمق=%d" % (name, len(cs), max(depths)))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
signed = cv2.contourArea(c, True)
peri = cv2.arcLength(c, True)
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
x, y, w, h = cv2.boundingRect(c)
print("المساحة: %.1f" % area)
print("المساحة الموجّهة: %.1f" % signed)
print("المحيط: %.2f" % peri)
print("المركز: (%.1f, %.1f)" % (cx, cy))
print("المستطيل المحيط: x=%d y=%d w=%d h=%d" % (x, y, w, h))
print("الاستطالة extent: %.3f" % (area / (w * h)))
print("الدائرية: %.3f" % (4 * np.pi * area / peri ** 2))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
circle = max([z for z in contours if cv2.contourArea(z) > 500],
             key=lambda z: 4 * np.pi * cv2.contourArea(z)
             / cv2.arcLength(z, True) ** 2)
peri = cv2.arcLength(circle, True)
area = cv2.contourArea(circle)
print("مساحة الدائرة الأصلية: %.1f" % area)
for eps in (0.005, 0.01, 0.02, 0.05):
    ap = cv2.approxPolyDP(circle, eps * peri, True)
    a = cv2.contourArea(ap)
    print("eps=%.3f رؤوس=%2d مساحة=%.1f خطأ=%.1f%%"
          # % (eps, len(ap), a, 100 * abs(a - area) / area))

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
hull = cv2.convexHull(c)
rect = cv2.minAreaRect(c)
(ccx, ccy), r = cv2.minEnclosingCircle(c)
print("مساحة الغلاف: %.1f" % cv2.contourArea(hull))
print("الصلابة: %.4f" % (area / cv2.contourArea(hull)))
print("أقل مستطيل: مركز=(%.1f, %.1f) حجم=(%.1f, %.1f) زاوية=%.1f"
      # % (rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]))
print("أصغر دائرة: نصف قطر=%.2f | نسبة الملء=%.3f"
      # % (r, area / (np.pi * r * r)))

circle = max([z for z in contours if 4 * np.pi * cv2.contourArea(z)
              / cv2.arcLength(z, True) ** 2 > 0.85],
             key=cv2.contourArea)
ell = cv2.fitEllipse(circle)
print("بيضوي الدائرة الكبيرة: محاور=(%.1f, %.1f) زاوية=%.1f"
      # % (ell[1][0], ell[1][1], ell[2]))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
def hu_log(c):
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    return [-np.sign(v) * np.log10(abs(v)) if v else 0.0 for v in hu]

shape = np.zeros((400, 400), np.uint8)
cv2.rectangle(shape, (100, 80), (300, 180), 255, -1)
cv2.rectangle(shape, (100, 180), (200, 320), 255, -1)
cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
L = max(cs, key=cv2.contourArea)
M = cv2.getRotationMatrix2D((200, 200), 37, 1)
rot = cv2.warpAffine(shape, M, (400, 400))
cs, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Lr = max(cs, key=cv2.contourArea)
a, b = hu_log(L), hu_log(Lr)
for i in range(7):
    print("h%d: %8.4f  %8.4f" % (i + 1, a[i], b[i]))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
rows = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    peri = cv2.arcLength(c, True)
    verts = len(cv2.approxPolyDP(c, 0.025 * peri, True))
    circ = 4 * np.pi * area / peri ** 2
    if verts == 3:
        label = "Triangle"
    elif verts == 4:
        label = "Rectangle"
    elif circ > 0.85:
        label = "Circle"
    else:
        label = "Unknown"
    M = cv2.moments(c)
    rows.append((M["m10"] / M["m00"], M["m01"] / M["m00"],
                 label, area, peri, verts, circ))

rows.sort(key=lambda r: (r[0], r[1]))
counts = {}
for cx, cy, label, area, peri, verts, circ in rows:
    counts[label] = counts.get(label, 0) + 1
    print("x=%3.0f y=%3.0f %-9s area=%6.0f p=%5.0f verts=%d circ=%.3f"
          # % (cx, cy, label, area, peri, verts, circ))
print("العدّ:", counts)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

scene = make_belt()
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("عدد الكونتورات:", len(contours))
print("شكل القناع:", bw.shape, "| النوع:", bw.dtype)
print("مجموع النقاط:", sum(len(c) for c in contours))
print("مساحة أكبرها: %.1f" % cv2.contourArea(max(contours, key=cv2.contourArea)))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
none, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
simple, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_none = sum(len(c) for c in none)
n_simple = sum(len(c) for c in simple)
print("NONE:", n_none, "| SIMPLE:", n_simple)
print("التوفير: %.1f%%" % (100 * (1 - n_simple / n_none)))
print("أطول كونتور NONE:", max(len(c) for c in none))
print("أطول كونتور SIMPLE:", max(len(c) for c in simple))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
m = np.zeros((240, 360), np.uint8)
cv2.rectangle(m, (30, 30), (200, 210), 255, -1)
cv2.rectangle(m, (60, 60), (170, 180), 0, -1)
cv2.circle(m, (115, 120), 30, 255, -1)
cv2.circle(m, (280, 120), 60, 255, -1)
cv2.circle(m, (280, 120), 25, 0, -1)

for name, mode in [("EXTERNAL", cv2.RETR_EXTERNAL),
                   ("LIST", cv2.RETR_LIST),
                   ("CCOMP", cv2.RETR_CCOMP),
                   ("TREE", cv2.RETR_TREE)]:
    cs, h = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    hh = h.reshape(-1, 4)
    depths = []
    for i in range(len(cs)):
        d, p = 0, int(hh[i, 3])
        while p != -1:
            d += 1
            p = int(hh[p, 3])
        depths.append(d)
    print("%-8s عدد=%d أقصى عمق=%d" % (name, len(cs), max(depths)))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
signed = cv2.contourArea(c, True)
peri = cv2.arcLength(c, True)
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
x, y, w, h = cv2.boundingRect(c)
print("المساحة: %.1f" % area)
print("المساحة الموجّهة: %.1f" % signed)
print("المحيط: %.2f" % peri)
print("المركز: (%.1f, %.1f)" % (cx, cy))
print("المستطيل المحيط: x=%d y=%d w=%d h=%d" % (x, y, w, h))
print("الاستطالة extent: %.3f" % (area / (w * h)))
print("الدائرية: %.3f" % (4 * np.pi * area / peri ** 2))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
circle = max([z for z in contours if cv2.contourArea(z) > 500],
             key=lambda z: 4 * np.pi * cv2.contourArea(z)
             / cv2.arcLength(z, True) ** 2)
peri = cv2.arcLength(circle, True)
area = cv2.contourArea(circle)
print("مساحة الدائرة الأصلية: %.1f" % area)
for eps in (0.005, 0.01, 0.02, 0.05):
    ap = cv2.approxPolyDP(circle, eps * peri, True)
    a = cv2.contourArea(ap)
    print("eps=%.3f رؤوس=%2d مساحة=%.1f خطأ=%.1f%%"
          # % (eps, len(ap), a, 100 * abs(a - area) / area))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
hull = cv2.convexHull(c)
rect = cv2.minAreaRect(c)
(ccx, ccy), r = cv2.minEnclosingCircle(c)
print("مساحة الغلاف: %.1f" % cv2.contourArea(hull))
print("الصلابة: %.4f" % (area / cv2.contourArea(hull)))
print("أقل مستطيل: مركز=(%.1f, %.1f) حجم=(%.1f, %.1f) زاوية=%.1f"
      # % (rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]))
print("أصغر دائرة: نصف قطر=%.2f | نسبة الملء=%.3f"
      # % (r, area / (np.pi * r * r)))

circle = max([z for z in contours if 4 * np.pi * cv2.contourArea(z)
              / cv2.arcLength(z, True) ** 2 > 0.85],
             key=cv2.contourArea)
ell = cv2.fitEllipse(circle)
print("بيضوي الدائرة الكبيرة: محاور=(%.1f, %.1f) زاوية=%.1f"
      # % (ell[1][0], ell[1][1], ell[2]))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def hu_log(c):
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    return [-np.sign(v) * np.log10(abs(v)) if v else 0.0 for v in hu]

shape = np.zeros((400, 400), np.uint8)
cv2.rectangle(shape, (100, 80), (300, 180), 255, -1)
cv2.rectangle(shape, (100, 180), (200, 320), 255, -1)
cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
L = max(cs, key=cv2.contourArea)
M = cv2.getRotationMatrix2D((200, 200), 37, 1)
rot = cv2.warpAffine(shape, M, (400, 400))
cs, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Lr = max(cs, key=cv2.contourArea)
a, b = hu_log(L), hu_log(Lr)
for i in range(7):
    print("h%d: %8.4f  %8.4f" % (i + 1, a[i], b[i]))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
rows = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    peri = cv2.arcLength(c, True)
    verts = len(cv2.approxPolyDP(c, 0.025 * peri, True))
    circ = 4 * np.pi * area / peri ** 2
    if verts == 3:
        label = "Triangle"
    elif verts == 4:
        label = "Rectangle"
    elif circ > 0.85:
        label = "Circle"
    else:
        label = "Unknown"
    M = cv2.moments(c)
    rows.append((M["m10"] / M["m00"], M["m01"] / M["m00"],
                 label, area, peri, verts, circ))

rows.sort(key=lambda r: (r[0], r[1]))
counts = {}
for cx, cy, label, area, peri, verts, circ in rows:
    counts[label] = counts.get(label, 0) + 1
    print("x=%3.0f y=%3.0f %-9s area=%6.0f p=%5.0f verts=%d circ=%.3f"
          # % (cx, cy, label, area, peri, verts, circ))
print("العدّ:", counts)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الكائنات المتصلة والكونتورات: قراءة الأشكال
OpenCV 5 and AI Computer Vision — Contours

Book slug : opencv-5-ai-vision
Chapter   : ch-10-ch10-contours (lesson 10)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

scene = make_belt()
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("عدد الكونتورات:", len(contours))
print("شكل القناع:", bw.shape, "| النوع:", bw.dtype)
print("مجموع النقاط:", sum(len(c) for c in contours))
print("مساحة أكبرها: %.1f" % cv2.contourArea(max(contours, key=cv2.contourArea)))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
none, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
simple, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_none = sum(len(c) for c in none)
n_simple = sum(len(c) for c in simple)
print("NONE:", n_none, "| SIMPLE:", n_simple)
print("التوفير: %.1f%%" % (100 * (1 - n_simple / n_none)))
print("أطول كونتور NONE:", max(len(c) for c in none))
print("أطول كونتور SIMPLE:", max(len(c) for c in simple))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
m = np.zeros((240, 360), np.uint8)
cv2.rectangle(m, (30, 30), (200, 210), 255, -1)
cv2.rectangle(m, (60, 60), (170, 180), 0, -1)
cv2.circle(m, (115, 120), 30, 255, -1)
cv2.circle(m, (280, 120), 60, 255, -1)
cv2.circle(m, (280, 120), 25, 0, -1)

for name, mode in [("EXTERNAL", cv2.RETR_EXTERNAL),
                   ("LIST", cv2.RETR_LIST),
                   ("CCOMP", cv2.RETR_CCOMP),
                   ("TREE", cv2.RETR_TREE)]:
    cs, h = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    hh = h.reshape(-1, 4)
    depths = []
    for i in range(len(cs)):
        d, p = 0, int(hh[i, 3])
        while p != -1:
            d += 1
            p = int(hh[p, 3])
        depths.append(d)
    print("%-8s عدد=%d أقصى عمق=%d" % (name, len(cs), max(depths)))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
signed = cv2.contourArea(c, True)
peri = cv2.arcLength(c, True)
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
x, y, w, h = cv2.boundingRect(c)
print("المساحة: %.1f" % area)
print("المساحة الموجّهة: %.1f" % signed)
print("المحيط: %.2f" % peri)
print("المركز: (%.1f, %.1f)" % (cx, cy))
print("المستطيل المحيط: x=%d y=%d w=%d h=%d" % (x, y, w, h))
print("الاستطالة extent: %.3f" % (area / (w * h)))
print("الدائرية: %.3f" % (4 * np.pi * area / peri ** 2))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
circle = max([z for z in contours if cv2.contourArea(z) > 500],
             key=lambda z: 4 * np.pi * cv2.contourArea(z)
             / cv2.arcLength(z, True) ** 2)
peri = cv2.arcLength(circle, True)
area = cv2.contourArea(circle)
print("مساحة الدائرة الأصلية: %.1f" % area)
for eps in (0.005, 0.01, 0.02, 0.05):
    ap = cv2.approxPolyDP(circle, eps * peri, True)
    a = cv2.contourArea(ap)
    print("eps=%.3f رؤوس=%2d مساحة=%.1f خطأ=%.1f%%"
          # % (eps, len(ap), a, 100 * abs(a - area) / area))

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
hull = cv2.convexHull(c)
rect = cv2.minAreaRect(c)
(ccx, ccy), r = cv2.minEnclosingCircle(c)
print("مساحة الغلاف: %.1f" % cv2.contourArea(hull))
print("الصلابة: %.4f" % (area / cv2.contourArea(hull)))
print("أقل مستطيل: مركز=(%.1f, %.1f) حجم=(%.1f, %.1f) زاوية=%.1f"
      # % (rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]))
print("أصغر دائرة: نصف قطر=%.2f | نسبة الملء=%.3f"
      # % (r, area / (np.pi * r * r)))

circle = max([z for z in contours if 4 * np.pi * cv2.contourArea(z)
              / cv2.arcLength(z, True) ** 2 > 0.85],
             key=cv2.contourArea)
ell = cv2.fitEllipse(circle)
print("بيضوي الدائرة الكبيرة: محاور=(%.1f, %.1f) زاوية=%.1f"
      # % (ell[1][0], ell[1][1], ell[2]))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
def hu_log(c):
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    return [-np.sign(v) * np.log10(abs(v)) if v else 0.0 for v in hu]

shape = np.zeros((400, 400), np.uint8)
cv2.rectangle(shape, (100, 80), (300, 180), 255, -1)
cv2.rectangle(shape, (100, 180), (200, 320), 255, -1)
cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
L = max(cs, key=cv2.contourArea)
M = cv2.getRotationMatrix2D((200, 200), 37, 1)
rot = cv2.warpAffine(shape, M, (400, 400))
cs, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Lr = max(cs, key=cv2.contourArea)
a, b = hu_log(L), hu_log(Lr)
for i in range(7):
    print("h%d: %8.4f  %8.4f" % (i + 1, a[i], b[i]))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch10_contours.md
# ────────────────────────────────────────────────────────────────────────
rows = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    peri = cv2.arcLength(c, True)
    verts = len(cv2.approxPolyDP(c, 0.025 * peri, True))
    circ = 4 * np.pi * area / peri ** 2
    if verts == 3:
        label = "Triangle"
    elif verts == 4:
        label = "Rectangle"
    elif circ > 0.85:
        label = "Circle"
    else:
        label = "Unknown"
    M = cv2.moments(c)
    rows.append((M["m10"] / M["m00"], M["m01"] / M["m00"],
                 label, area, peri, verts, circ))

rows.sort(key=lambda r: (r[0], r[1]))
counts = {}
for cx, cy, label, area, peri, verts, circ in rows:
    counts[label] = counts.get(label, 0) + 1
    print("x=%3.0f y=%3.0f %-9s area=%6.0f p=%5.0f verts=%d circ=%.3f"
          # % (cx, cy, label, area, peri, verts, circ))
print("العدّ:", counts)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

scene = make_belt()
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("عدد الكونتورات:", len(contours))
print("شكل القناع:", bw.shape, "| النوع:", bw.dtype)
print("مجموع النقاط:", sum(len(c) for c in contours))
print("مساحة أكبرها: %.1f" % cv2.contourArea(max(contours, key=cv2.contourArea)))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
none, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
simple, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_none = sum(len(c) for c in none)
n_simple = sum(len(c) for c in simple)
print("NONE:", n_none, "| SIMPLE:", n_simple)
print("التوفير: %.1f%%" % (100 * (1 - n_simple / n_none)))
print("أطول كونتور NONE:", max(len(c) for c in none))
print("أطول كونتور SIMPLE:", max(len(c) for c in simple))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
m = np.zeros((240, 360), np.uint8)
cv2.rectangle(m, (30, 30), (200, 210), 255, -1)
cv2.rectangle(m, (60, 60), (170, 180), 0, -1)
cv2.circle(m, (115, 120), 30, 255, -1)
cv2.circle(m, (280, 120), 60, 255, -1)
cv2.circle(m, (280, 120), 25, 0, -1)

for name, mode in [("EXTERNAL", cv2.RETR_EXTERNAL),
                   ("LIST", cv2.RETR_LIST),
                   ("CCOMP", cv2.RETR_CCOMP),
                   ("TREE", cv2.RETR_TREE)]:
    cs, h = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
    hh = h.reshape(-1, 4)
    depths = []
    for i in range(len(cs)):
        d, p = 0, int(hh[i, 3])
        while p != -1:
            d += 1
            p = int(hh[p, 3])
        depths.append(d)
    print("%-8s عدد=%d أقصى عمق=%d" % (name, len(cs), max(depths)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
signed = cv2.contourArea(c, True)
peri = cv2.arcLength(c, True)
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
x, y, w, h = cv2.boundingRect(c)
print("المساحة: %.1f" % area)
print("المساحة الموجّهة: %.1f" % signed)
print("المحيط: %.2f" % peri)
print("المركز: (%.1f, %.1f)" % (cx, cy))
print("المستطيل المحيط: x=%d y=%d w=%d h=%d" % (x, y, w, h))
print("الاستطالة extent: %.3f" % (area / (w * h)))
print("الدائرية: %.3f" % (4 * np.pi * area / peri ** 2))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
circle = max([z for z in contours if cv2.contourArea(z) > 500],
             key=lambda z: 4 * np.pi * cv2.contourArea(z)
             / cv2.arcLength(z, True) ** 2)
peri = cv2.arcLength(circle, True)
area = cv2.contourArea(circle)
print("مساحة الدائرة الأصلية: %.1f" % area)
for eps in (0.005, 0.01, 0.02, 0.05):
    ap = cv2.approxPolyDP(circle, eps * peri, True)
    a = cv2.contourArea(ap)
    print("eps=%.3f رؤوس=%2d مساحة=%.1f خطأ=%.1f%%"
          # % (eps, len(ap), a, 100 * abs(a - area) / area))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
c = max(contours, key=cv2.contourArea)
area = cv2.contourArea(c)
hull = cv2.convexHull(c)
rect = cv2.minAreaRect(c)
(ccx, ccy), r = cv2.minEnclosingCircle(c)
print("مساحة الغلاف: %.1f" % cv2.contourArea(hull))
print("الصلابة: %.4f" % (area / cv2.contourArea(hull)))
print("أقل مستطيل: مركز=(%.1f, %.1f) حجم=(%.1f, %.1f) زاوية=%.1f"
      # % (rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]))
print("أصغر دائرة: نصف قطر=%.2f | نسبة الملء=%.3f"
      # % (r, area / (np.pi * r * r)))

circle = max([z for z in contours if 4 * np.pi * cv2.contourArea(z)
              / cv2.arcLength(z, True) ** 2 > 0.85],
             key=cv2.contourArea)
ell = cv2.fitEllipse(circle)
print("بيضوي الدائرة الكبيرة: محاور=(%.1f, %.1f) زاوية=%.1f"
      # % (ell[1][0], ell[1][1], ell[2]))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def hu_log(c):
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    return [-np.sign(v) * np.log10(abs(v)) if v else 0.0 for v in hu]

shape = np.zeros((400, 400), np.uint8)
cv2.rectangle(shape, (100, 80), (300, 180), 255, -1)
cv2.rectangle(shape, (100, 180), (200, 320), 255, -1)
cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
L = max(cs, key=cv2.contourArea)
M = cv2.getRotationMatrix2D((200, 200), 37, 1)
rot = cv2.warpAffine(shape, M, (400, 400))
cs, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Lr = max(cs, key=cv2.contourArea)
a, b = hu_log(L), hu_log(Lr)
for i in range(7):
    print("h%d: %8.4f  %8.4f" % (i + 1, a[i], b[i]))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-10-ch10-contours.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
rows = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    peri = cv2.arcLength(c, True)
    verts = len(cv2.approxPolyDP(c, 0.025 * peri, True))
    circ = 4 * np.pi * area / peri ** 2
    if verts == 3:
        label = "Triangle"
    elif verts == 4:
        label = "Rectangle"
    elif circ > 0.85:
        label = "Circle"
    else:
        label = "Unknown"
    M = cv2.moments(c)
    rows.append((M["m10"] / M["m00"], M["m01"] / M["m00"],
                 label, area, peri, verts, circ))

rows.sort(key=lambda r: (r[0], r[1]))
counts = {}
for cx, cy, label, area, peri, verts, circ in rows:
    counts[label] = counts.get(label, 0) + 1
    print("x=%3.0f y=%3.0f %-9s area=%6.0f p=%5.0f verts=%d circ=%.3f"
          # % (cx, cy, label, area, peri, verts, circ))
print("العدّ:", counts)
