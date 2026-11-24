#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأداء والتحسين: من بايثون البطيء إلى خط إنتاج حيّ
OpenCV 5 and AI Computer Vision — Perf

Book slug : opencv-5-ai-vision
Chapter   : ch-19-ch19-perf (lesson 19)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-19-ch19-perf.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأداء والتحسين: من بايثون البطيء إلى خط إنتاج حيّ
OpenCV 5 and AI Computer Vision — Perf

Book slug : opencv-5-ai-vision
Chapter   : ch-19-ch19-perf (lesson 19)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch19_perf.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

t0 = cv2.getTickCount()          # أو time.perf_counter()
 # الخطوة المرشحة للقياس
dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
print("زمن الخطوة:", round(dt * 1000, 2), "ms")
print("معدل مكافئ:", round(1.0 / dt, 1), "fps")

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch19_perf.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# بطيء جداً (مثال تعليمي فقط):
h, w = gray.shape
for y in range(h):
    for x in range(w):
        inv[y, x] = 255 - gray[y, x]

# أسرع بمئات المرات:
inv = 255 - gray                  # دعم NumPy على كامل المصفوفة
binary = (gray > 128)             # المنطق يعمل على المصفوفة لا على البكسل

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch19_perf.md
# ────────────────────────────────────────────────────────────────────────
small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)      # نصف الحجم، أربعة مرات أسرع
fast  = cv2.Sobel(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                 cv2.CV_64F, 1, 0)

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch19_perf.md
# ────────────────────────────────────────────────────────────────────────
# قراءة مسبقة: احجز مسبقاً (pre-allocated) مصفوفة المخرجات
out = np.empty_like(gray)
cv2.GaussianBlur(gray, (7, 7), 1.4, dst=out)

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch19_perf.md
# ────────────────────────────────────────────────────────────────────────
import threading
import queue

frames_q = queue.Queue(maxsize=4)     # حاجز صغير (buffer)

def reader():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frames_q.full():
            frames_q.get()            # رمي الإطار الأقدم إذا تمتلئ
        frames_q.put(frame)

threading.Thread(target=reader, daemon=True).start()

# المعالجة في الخيط الرئيسي
while True:
    frame = frames_q.get()
    # ... التحليل هنا (العرض والمعالجة أصبحت مستقلة عن زمن القراءة)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-19-ch19-perf.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

t0 = cv2.getTickCount()          # أو time.perf_counter()
 # الخطوة المرشحة للقياس
dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
print("زمن الخطوة:", round(dt * 1000, 2), "ms")
print("معدل مكافئ:", round(1.0 / dt, 1), "fps")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-19-ch19-perf.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# بطيء جداً (مثال تعليمي فقط):
h, w = gray.shape
for y in range(h):
    for x in range(w):
        inv[y, x] = 255 - gray[y, x]

# أسرع بمئات المرات:
inv = 255 - gray                  # دعم NumPy على كامل المصفوفة
binary = (gray > 128)             # المنطق يعمل على المصفوفة لا على البكسل

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-19-ch19-perf.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)      # نصف الحجم، أربعة مرات أسرع
fast  = cv2.Sobel(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                 cv2.CV_64F, 1, 0)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-19-ch19-perf.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# قراءة مسبقة: احجز مسبقاً (pre-allocated) مصفوفة المخرجات
out = np.empty_like(gray)
cv2.GaussianBlur(gray, (7, 7), 1.4, dst=out)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-19-ch19-perf.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import threading
import queue

frames_q = queue.Queue(maxsize=4)     # حاجز صغير (buffer)

def reader():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frames_q.full():
            frames_q.get()            # رمي الإطار الأقدم إذا تمتلئ
        frames_q.put(frame)

threading.Thread(target=reader, daemon=True).start()

# المعالجة في الخيط الرئيسي
while True:
    frame = frames_q.get()
    # ... التحليل هنا (العرض والمعالجة أصبحت مستقلة عن زمن القراءة)
