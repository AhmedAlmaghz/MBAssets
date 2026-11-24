#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التتبع وطرح الخلفية: من التدفق الضوئي إلى التتبّع الحيّ
OpenCV 5 and AI Computer Vision — Tracking

Book slug : opencv-5-ai-vision
Chapter   : ch-18-ch18-tracking (lesson 18)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-18-ch18-tracking.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التتبع وطرح الخلفية: من التدفق الضوئي إلى التتبّع الحيّ
OpenCV 5 and AI Computer Vision — Tracking

Book slug : opencv-5-ai-vision
Chapter   : ch-18-ch18-tracking (lesson 18)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch18_tracking.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture("camera_feed.mp4")
bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25,
                                        detectShadows=True)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    mask = bs.apply(frame)
    # تنظيف الماسك: فتح + غلق (من الفصل 9)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch18_tracking.md
# ────────────────────────────────────────────────────────────────────────
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
objects = []
for c in cnts:
    a = cv2.contourArea(c)
    if a < 800:                        # تجاهل النقاط الصغيرة (ضجيج)
        continue
    x, y, w, h = cv2.boundingRect(c)
    objects.append((x, y, w, h))

for (x, y, w, h) in objects:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch18_tracking.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old = cap.read()
gray_old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(gray_old, maxCorners=100, qualityLevel=0.3,
                              minDistance=7, blockSize=7)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None,
                                           **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for (x1, y1), (x2, y2) in zip(good_old, good_new):
        dx, dy = float(x2 - x1), float(y2 - y1)
        speed = (dx ** 2 + dy ** 2) ** 0.5
        cv2.arrowedLine(frame, (int(x1), int(y1)),
                        (int(x2), int(y2)), (0, 255, 0), 2)
    p0 = good_new.reshape(-1, 1, 2)
    gray_old = gray_new

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch18_tracking.md
# ────────────────────────────────────────────────────────────────────────
tracker = cv2.TrackerCSRT_create()
ret, frame = cap.read()
box = (240, 130, 120, 220)             # مختار يدوياً أو من مخرج الكشف (DNN)
tracker.init(frame, box)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-18-ch18-tracking.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture("camera_feed.mp4")
bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25,
                                        detectShadows=True)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    mask = bs.apply(frame)
    # تنظيف الماسك: فتح + غلق (من الفصل 9)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-18-ch18-tracking.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
objects = []
for c in cnts:
    a = cv2.contourArea(c)
    if a < 800:                        # تجاهل النقاط الصغيرة (ضجيج)
        continue
    x, y, w, h = cv2.boundingRect(c)
    objects.append((x, y, w, h))

for (x, y, w, h) in objects:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-18-ch18-tracking.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old = cap.read()
gray_old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(gray_old, maxCorners=100, qualityLevel=0.3,
                              minDistance=7, blockSize=7)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None,
                                           **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for (x1, y1), (x2, y2) in zip(good_old, good_new):
        dx, dy = float(x2 - x1), float(y2 - y1)
        speed = (dx ** 2 + dy ** 2) ** 0.5
        cv2.arrowedLine(frame, (int(x1), int(y1)),
                        (int(x2), int(y2)), (0, 255, 0), 2)
    p0 = good_new.reshape(-1, 1, 2)
    gray_old = gray_new

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-18-ch18-tracking.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
tracker = cv2.TrackerCSRT_create()
ret, frame = cap.read()
box = (240, 130, 120, 220)             # مختار يدوياً أو من مخرج الكشف (DNN)
tracker.init(frame, box)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
