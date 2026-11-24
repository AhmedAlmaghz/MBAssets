#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — مطابقة القوالب: إيجاد الجسم المعروف
OpenCV 5 and AI Computer Vision — Template

Book slug : opencv-5-ai-vision
Chapter   : ch-13-ch13-template (lesson 13)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-13-ch13-template.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — مطابقة القوالب: إيجاد الجسم المعروف
OpenCV 5 and AI Computer Vision — Template

Book slug : opencv-5-ai-vision
Chapter   : ch-13-ch13-template (lesson 13)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch13_template.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img  = cv2.imread("assets/samples/messi5.jpg", cv2.IMREAD_GRAYSCALE)
# قارئ صغير: نقطعه من الصورة نفسها للتجربة (منطق العمل الحقيقي: قالب محفوظ)
tpl  = img[80:140, 210:280]

res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print("أفضل تطابق عند:", max_loc, "بدرجة:", max_val)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch13_template.md
# ────────────────────────────────────────────────────────────────────────
img_color = cv2.imread("assets/samples/box.png")

# نستخرج قارئ صورة box_in_scene من مثال المكتبة
scene = cv2.imread("assets/samples/box_in_scene.png", 0)
tpl  = cv2.imread("assets/samples/box.png", 0)

res  = cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED)
loc  = np.where(res >= 0.75)                     # كل النقاط فوق العتبة

matches = []
th, tw = tpl.shape[0:2]
for pt in zip(*loc[::-1]):                       # (x, y)
    matches.append(pt)

# إزالة التكرار (NMS مبسط): نقطة «عربية» لنفس الجسم
picked = []
for pt in sorted(matches):
    if all(abs(pt[0]-q[0]) > tw*0.3 or abs(pt[1]-q[1]) > th*0.3
           for q in picked):
        picked.append(pt)

for (x, y) in picked:
    cv2.rectangle(img_color, (x, y), (x+tw, y+th), (0, 255, 0), 2)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-13-ch13-template.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img  = cv2.imread("assets/samples/messi5.jpg", cv2.IMREAD_GRAYSCALE)
# قارئ صغير: نقطعه من الصورة نفسها للتجربة (منطق العمل الحقيقي: قالب محفوظ)
tpl  = img[80:140, 210:280]

res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print("أفضل تطابق عند:", max_loc, "بدرجة:", max_val)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-13-ch13-template.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
img_color = cv2.imread("assets/samples/box.png")

# نستخرج قارئ صورة box_in_scene من مثال المكتبة
scene = cv2.imread("assets/samples/box_in_scene.png", 0)
tpl  = cv2.imread("assets/samples/box.png", 0)

res  = cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED)
loc  = np.where(res >= 0.75)                     # كل النقاط فوق العتبة

matches = []
th, tw = tpl.shape[0:2]
for pt in zip(*loc[::-1]):                       # (x, y)
    matches.append(pt)

# إزالة التكرار (NMS مبسط): نقطة «عربية» لنفس الجسم
picked = []
for pt in sorted(matches):
    if all(abs(pt[0]-q[0]) > tw*0.3 or abs(pt[1]-q[1]) > th*0.3
           for q in picked):
        picked.append(pt)

for (x, y) in picked:
    cv2.rectangle(img_color, (x, y), (x+tw, y+th), (0, 255, 0), 2)
