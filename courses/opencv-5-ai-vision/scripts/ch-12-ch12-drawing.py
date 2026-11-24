#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الرسم والتراكب والنصوص: توثيق المخرجات
OpenCV 5 and AI Computer Vision — Drawing

Book slug : opencv-5-ai-vision
Chapter   : ch-12-ch12-drawing (lesson 12)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-12-ch12-drawing.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الرسم والتراكب والنصوص: توثيق المخرجات
OpenCV 5 and AI Computer Vision — Drawing

Book slug : opencv-5-ai-vision
Chapter   : ch-12-ch12-drawing (lesson 12)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch12_drawing.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
canvas = img.copy()

cv2.line(canvas, (60, 60), (250, 100), (0, 255, 0), 2)          # خط
cv2.rectangle(canvas, (150, 180), (320, 340), (255, 0, 0), 2)   # مستطيل
cv2.rectangle(canvas, (380, 60), (480, 160), (0, 0, 255), -1)   # filled rect
cv2.circle(canvas, (250, 250), 60, (0, 255, 255), 2)            # دائرة
cv2.ellipse(canvas, (400, 260), (90, 50), -30, 0, 360,
            (255, 255, 0), 2)                                    # بيضوي
pts = np.array([[120, 40], [180, 40], [150, 90]], np.int32)
cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)                # مثلث مكشوف
cv2.fillPoly(canvas, [pts], (200, 120, 60))                     # مثلث ممتلئ

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch12_drawing.md
# ────────────────────────────────────────────────────────────────────────
cv2.putText(canvas, "Board: 3", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            cv2.LINE_AA)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch12_drawing.md
# ────────────────────────────────────────────────────────────────────────
# pip install arabic-reshaper python-bidi
import arabic_reshaper
from bidi.algorithm import get_display

def draw_text(img, text, org, size=0.9, color=(255, 255, 255), thick=2):
    txt = get_display(arabic_reshaper.reshape(text))
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thick, cv2.LINE_AA)
    return img

canvas = draw_text(canvas, "مستطيل: مساحة 12,450", (150, 165))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch12_drawing.md
# ────────────────────────────────────────────────────────────────────────
overlay = canvas.copy()
cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 60),
              (15, 76, 129), -1)                 # BGR أزرق داكن
alpha = 0.6
canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

draw_text(canvas, "نظام مراقبة الإنتاج — 2026", (10, 40),
          color=(255, 255, 255))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch12_drawing.md
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)

report = img.copy()
overlay = report.copy()
cv2.rectangle(overlay, (0, 0), (report.shape[1], 44), (20, 70, 120), -1)
report = cv2.addWeighted(overlay, 0.6, report, 0.4, 0)

total_area = 0
for c in cnts:
    a = cv2.contourArea(c)
    if a < 300:
        continue
    total_area += a
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(report, (x, y), (x + w, y + h), (0, 200, 0), 2)
    txt = "Area: %d" % int(a)
    cv2.putText(report, txt, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

cv2.putText(report, "Objects: %d | Total area: %d"
            # % (len(cnts), int(total_area)), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA)

cv2.imwrite("visual_report.png", report)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-12-ch12-drawing.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
canvas = img.copy()

cv2.line(canvas, (60, 60), (250, 100), (0, 255, 0), 2)          # خط
cv2.rectangle(canvas, (150, 180), (320, 340), (255, 0, 0), 2)   # مستطيل
cv2.rectangle(canvas, (380, 60), (480, 160), (0, 0, 255), -1)   # filled rect
cv2.circle(canvas, (250, 250), 60, (0, 255, 255), 2)            # دائرة
cv2.ellipse(canvas, (400, 260), (90, 50), -30, 0, 360,
            (255, 255, 0), 2)                                    # بيضوي
pts = np.array([[120, 40], [180, 40], [150, 90]], np.int32)
cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)                # مثلث مكشوف
cv2.fillPoly(canvas, [pts], (200, 120, 60))                     # مثلث ممتلئ

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-12-ch12-drawing.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
cv2.putText(canvas, "Board: 3", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            cv2.LINE_AA)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-12-ch12-drawing.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# pip install arabic-reshaper python-bidi
import arabic_reshaper
from bidi.algorithm import get_display

def draw_text(img, text, org, size=0.9, color=(255, 255, 255), thick=2):
    txt = get_display(arabic_reshaper.reshape(text))
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thick, cv2.LINE_AA)
    return img

canvas = draw_text(canvas, "مستطيل: مساحة 12,450", (150, 165))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-12-ch12-drawing.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
overlay = canvas.copy()
cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 60),
              (15, 76, 129), -1)                 # BGR أزرق داكن
alpha = 0.6
canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

draw_text(canvas, "نظام مراقبة الإنتاج — 2026", (10, 40),
          color=(255, 255, 255))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-12-ch12-drawing.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)

report = img.copy()
overlay = report.copy()
cv2.rectangle(overlay, (0, 0), (report.shape[1], 44), (20, 70, 120), -1)
report = cv2.addWeighted(overlay, 0.6, report, 0.4, 0)

total_area = 0
for c in cnts:
    a = cv2.contourArea(c)
    if a < 300:
        continue
    total_area += a
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(report, (x, y), (x + w, y + h), (0, 200, 0), 2)
    txt = "Area: %d" % int(a)
    cv2.putText(report, txt, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

cv2.putText(report, "Objects: %d | Total area: %d"
            # % (len(cnts), int(total_area)), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA)

cv2.imwrite("visual_report.png", report)
