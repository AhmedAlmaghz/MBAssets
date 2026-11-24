#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأدوات المصرفية للصور: القراءة، الكتابة، العرض، والفيديو
OpenCV 5 and AI Computer Vision — Io Video

Book slug : opencv-5-ai-vision
Chapter   : ch-03-ch03-io-video (lesson 3)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-03-ch03-io-video.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأدوات المصرفية للصور: القراءة، الكتابة، العرض، والفيديو
OpenCV 5 and AI Computer Vision — Io Video

Book slug : opencv-5-ai-vision
Chapter   : ch-03-ch03-io-video (lesson 3)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-03-ch03-io-video.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأدوات المصرفية للصور: القراءة، الكتابة، العرض، والفيديو
OpenCV 5 and AI Computer Vision — Io Video

Book slug : opencv-5-ai-vision
Chapter   : ch-03-ch03-io-video (lesson 3)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2

PATH = "assets/samples/lena.jpg"
img_color = cv2.imread(PATH, cv2.IMREAD_COLOR)        # BGR، ثلاث قنوات
img_gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)     # قناة واحدة
img_unchanged = cv2.imread(PATH, cv2.IMREAD_UNCHANGED) # كما في الملف

assert img_color is not None, "فشلت القراءة: راجع المسار"

print("color:", img_color.shape, img_color.dtype)
print("gray :", img_gray.shape, img_gray.dtype)
print("unch :", img_unchanged.shape, img_unchanged.dtype)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import os
import cv2

path = "assets/samples/lena.jpg"
print("هل الملف موجود؟", os.path.exists(path))
img = cv2.imread(path)
print("هل نجحت القراءة؟", img is not None)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import cv2

img = cv2.imread("assets/samples/lena.jpg")

for name, ext, params in [
    ("PNG", ".png", []),
    ("JPEG q95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ("JPEG q75", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 75]),
    ("JPEG q40", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 40]),
]:
    path = "out_" + name.replace(" ", "_") + ext
    cv2.imwrite(path, img, params)
    back = cv2.imread(path)
    same = np.array_equal(img, back)
    psnr = "∞" if same else "%.2f dB" % cv2.PSNR(img, back)
    print("%-8s %7.1f KB | PSNR: %s" % (name, os.path.getsize(path) / 1024, psnr))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")

# في بيئة بسطح مكتب:
# cv2.imshow("lena", img)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()

# في بيئة بلا شاشة (خادم/دفتر):
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig("preview.png", dpi=100)
print("حُفظ العرض في preview.png")

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
while True:
    # ... معالجة الإطار ...
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2

source = "assets/samples/demo.mp4"   # أو 0 للكاميرا، أو رابط RTSP
cap = cv2.VideoCapture(source)

print("فُتح المصدر؟", cap.isOpened())
print("الدقة:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "×",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("معدل الإطارات المصدر:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

img = cv2.imread("assets/samples/lena.jpg")
base = cv2.resize(img, (320, 240))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("demo.mp4", fourcc, 30.0, (320, 240))
print("فُتح الكاتب؟", writer.isOpened())

for i in range(90):
    frame = base.copy()
    x = 10 + i * 3
    cv2.rectangle(frame, (x, 10), (x + 30, 40), (0, 255, 0), -1)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture("demo.mp4")
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
cap.release()

print("عدد الإطارات المقروءة:", count)
print("حجم الملف: %.1f KB" % (os.path.getsize("demo.mp4") / 1024))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

cap = cv2.VideoCapture("demo.mp4")
os.makedirs("frames", exist_ok=True)
idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % 10 == 0:
        cv2.imwrite("frames/frame_%03d.png" % idx, frame)
        saved += 1
    idx += 1
cap.release()
print("حُفظ", saved, "إطاراً من أصل", idx)

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import time
import cv2

img = cv2.imread("assets/samples/lena.jpg")
probe = cv2.resize(img, (320, 240))
probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

def timeit(fn, n=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000

ops = [
    ("تنعيم 7×7", lambda: cv2.GaussianBlur(probe, (7, 7), 0)),
    ("تحويل رمادي", lambda: cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)),
    ("حواف Canny", lambda: cv2.Canny(probe_gray, 100, 200)),
    ("ترميز JPEG", lambda: cv2.imencode(".jpg", probe, [cv2.IMWRITE_JPEG_QUALITY, 90])),
]
for name, fn in ops:
    ms = timeit(fn)
    print("%-12s %5.2f ms/إطار | %5.0f إطار/ثانية" % (name, ms, 1000 / ms))

# ────────────────────────────────────────────────────────────────────────
# [010] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
last = time.time()
count = 0
fps = 0.0
# داخل حلقة الفيديو:
count += 1
now = time.time()
if now - last >= 1.0:
    fps = count / (now - last)
    count, last = 0, now

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2

PATH = "assets/samples/lena.jpg"
img_color = cv2.imread(PATH, cv2.IMREAD_COLOR)        # BGR، ثلاث قنوات
img_gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)     # قناة واحدة
img_unchanged = cv2.imread(PATH, cv2.IMREAD_UNCHANGED) # كما في الملف

assert img_color is not None, "فشلت القراءة: راجع المسار"

print("color:", img_color.shape, img_color.dtype)
print("gray :", img_gray.shape, img_gray.dtype)
print("unch :", img_unchanged.shape, img_unchanged.dtype)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import cv2

path = "assets/samples/lena.jpg"
print("هل الملف موجود؟", os.path.exists(path))
img = cv2.imread(path)
print("هل نجحت القراءة؟", img is not None)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import cv2

img = cv2.imread("assets/samples/lena.jpg")

for name, ext, params in [
    ("PNG", ".png", []),
    ("JPEG q95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ("JPEG q75", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 75]),
    ("JPEG q40", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 40]),
]:
    path = "out_" + name.replace(" ", "_") + ext
    cv2.imwrite(path, img, params)
    back = cv2.imread(path)
    same = np.array_equal(img, back)
    psnr = "∞" if same else "%.2f dB" % cv2.PSNR(img, back)
    print("%-8s %7.1f KB | PSNR: %s" % (name, os.path.getsize(path) / 1024, psnr))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")

# في بيئة بسطح مكتب:
# cv2.imshow("lena", img)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()

# في بيئة بلا شاشة (خادم/دفتر):
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig("preview.png", dpi=100)
print("حُفظ العرض في preview.png")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
while True:
    # ... معالجة الإطار ...
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2

source = "assets/samples/demo.mp4"   # أو 0 للكاميرا، أو رابط RTSP
cap = cv2.VideoCapture(source)

print("فُتح المصدر؟", cap.isOpened())
print("الدقة:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "×",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("معدل الإطارات المصدر:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

img = cv2.imread("assets/samples/lena.jpg")
base = cv2.resize(img, (320, 240))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("demo.mp4", fourcc, 30.0, (320, 240))
print("فُتح الكاتب؟", writer.isOpened())

for i in range(90):
    frame = base.copy()
    x = 10 + i * 3
    cv2.rectangle(frame, (x, 10), (x + 30, 40), (0, 255, 0), -1)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture("demo.mp4")
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
cap.release()

print("عدد الإطارات المقروءة:", count)
print("حجم الملف: %.1f KB" % (os.path.getsize("demo.mp4") / 1024))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

cap = cv2.VideoCapture("demo.mp4")
os.makedirs("frames", exist_ok=True)
idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % 10 == 0:
        cv2.imwrite("frames/frame_%03d.png" % idx, frame)
        saved += 1
    idx += 1
cap.release()
print("حُفظ", saved, "إطاراً من أصل", idx)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import time
import cv2

img = cv2.imread("assets/samples/lena.jpg")
probe = cv2.resize(img, (320, 240))
probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

def timeit(fn, n=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000

ops = [
    ("تنعيم 7×7", lambda: cv2.GaussianBlur(probe, (7, 7), 0)),
    ("تحويل رمادي", lambda: cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)),
    ("حواف Canny", lambda: cv2.Canny(probe_gray, 100, 200)),
    ("ترميز JPEG", lambda: cv2.imencode(".jpg", probe, [cv2.IMWRITE_JPEG_QUALITY, 90])),
]
for name, fn in ops:
    ms = timeit(fn)
    print("%-12s %5.2f ms/إطار | %5.0f إطار/ثانية" % (name, ms, 1000 / ms))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
last = time.time()
count = 0
fps = 0.0
# داخل حلقة الفيديو:
count += 1
now = time.time()
if now - last >= 1.0:
    fps = count / (now - last)
    count, last = 0, now

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الأدوات المصرفية للصور: القراءة، الكتابة، العرض، والفيديو
OpenCV 5 and AI Computer Vision — Io Video

Book slug : opencv-5-ai-vision
Chapter   : ch-03-ch03-io-video (lesson 3)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2

PATH = "assets/samples/lena.jpg"
img_color = cv2.imread(PATH, cv2.IMREAD_COLOR)        # BGR، ثلاث قنوات
img_gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)     # قناة واحدة
img_unchanged = cv2.imread(PATH, cv2.IMREAD_UNCHANGED) # كما في الملف

assert img_color is not None, "فشلت القراءة: راجع المسار"

print("color:", img_color.shape, img_color.dtype)
print("gray :", img_gray.shape, img_gray.dtype)
print("unch :", img_unchanged.shape, img_unchanged.dtype)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import os
import cv2

path = "assets/samples/lena.jpg"
print("هل الملف موجود؟", os.path.exists(path))
img = cv2.imread(path)
print("هل نجحت القراءة؟", img is not None)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import cv2

img = cv2.imread("assets/samples/lena.jpg")

for name, ext, params in [
    ("PNG", ".png", []),
    ("JPEG q95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ("JPEG q75", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 75]),
    ("JPEG q40", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 40]),
]:
    path = "out_" + name.replace(" ", "_") + ext
    cv2.imwrite(path, img, params)
    back = cv2.imread(path)
    same = np.array_equal(img, back)
    psnr = "∞" if same else "%.2f dB" % cv2.PSNR(img, back)
    print("%-8s %7.1f KB | PSNR: %s" % (name, os.path.getsize(path) / 1024, psnr))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")

# في بيئة بسطح مكتب:
# cv2.imshow("lena", img)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()

# في بيئة بلا شاشة (خادم/دفتر):
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig("preview.png", dpi=100)
print("حُفظ العرض في preview.png")

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
while True:
    # ... معالجة الإطار ...
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2

source = "assets/samples/demo.mp4"   # أو 0 للكاميرا، أو رابط RTSP
cap = cv2.VideoCapture(source)

print("فُتح المصدر؟", cap.isOpened())
print("الدقة:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "×",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("معدل الإطارات المصدر:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

img = cv2.imread("assets/samples/lena.jpg")
base = cv2.resize(img, (320, 240))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("demo.mp4", fourcc, 30.0, (320, 240))
print("فُتح الكاتب؟", writer.isOpened())

for i in range(90):
    frame = base.copy()
    x = 10 + i * 3
    cv2.rectangle(frame, (x, 10), (x + 30, 40), (0, 255, 0), -1)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture("demo.mp4")
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
cap.release()

print("عدد الإطارات المقروءة:", count)
print("حجم الملف: %.1f KB" % (os.path.getsize("demo.mp4") / 1024))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

cap = cv2.VideoCapture("demo.mp4")
os.makedirs("frames", exist_ok=True)
idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % 10 == 0:
        cv2.imwrite("frames/frame_%03d.png" % idx, frame)
        saved += 1
    idx += 1
cap.release()
print("حُفظ", saved, "إطاراً من أصل", idx)

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
import time
import cv2

img = cv2.imread("assets/samples/lena.jpg")
probe = cv2.resize(img, (320, 240))
probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

def timeit(fn, n=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000

ops = [
    ("تنعيم 7×7", lambda: cv2.GaussianBlur(probe, (7, 7), 0)),
    ("تحويل رمادي", lambda: cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)),
    ("حواف Canny", lambda: cv2.Canny(probe_gray, 100, 200)),
    ("ترميز JPEG", lambda: cv2.imencode(".jpg", probe, [cv2.IMWRITE_JPEG_QUALITY, 90])),
]
for name, fn in ops:
    ms = timeit(fn)
    print("%-12s %5.2f ms/إطار | %5.0f إطار/ثانية" % (name, ms, 1000 / ms))

# ────────────────────────────────────────────────────────────────────────
# [010] content/chapters/ch03_io_video.md
# ────────────────────────────────────────────────────────────────────────
last = time.time()
count = 0
fps = 0.0
# داخل حلقة الفيديو:
count += 1
now = time.time()
if now - last >= 1.0:
    fps = count / (now - last)
    count, last = 0, now

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2

PATH = "assets/samples/lena.jpg"
img_color = cv2.imread(PATH, cv2.IMREAD_COLOR)        # BGR، ثلاث قنوات
img_gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)     # قناة واحدة
img_unchanged = cv2.imread(PATH, cv2.IMREAD_UNCHANGED) # كما في الملف

assert img_color is not None, "فشلت القراءة: راجع المسار"

print("color:", img_color.shape, img_color.dtype)
print("gray :", img_gray.shape, img_gray.dtype)
print("unch :", img_unchanged.shape, img_unchanged.dtype)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
import cv2

path = "assets/samples/lena.jpg"
print("هل الملف موجود؟", os.path.exists(path))
img = cv2.imread(path)
print("هل نجحت القراءة؟", img is not None)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import cv2

img = cv2.imread("assets/samples/lena.jpg")

for name, ext, params in [
    ("PNG", ".png", []),
    ("JPEG q95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ("JPEG q75", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 75]),
    ("JPEG q40", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 40]),
]:
    path = "out_" + name.replace(" ", "_") + ext
    cv2.imwrite(path, img, params)
    back = cv2.imread(path)
    same = np.array_equal(img, back)
    psnr = "∞" if same else "%.2f dB" % cv2.PSNR(img, back)
    print("%-8s %7.1f KB | PSNR: %s" % (name, os.path.getsize(path) / 1024, psnr))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")

# في بيئة بسطح مكتب:
# cv2.imshow("lena", img)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()

# في بيئة بلا شاشة (خادم/دفتر):
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig("preview.png", dpi=100)
print("حُفظ العرض في preview.png")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
while True:
    # ... معالجة الإطار ...
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2

source = "assets/samples/demo.mp4"   # أو 0 للكاميرا، أو رابط RTSP
cap = cv2.VideoCapture(source)

print("فُتح المصدر؟", cap.isOpened())
print("الدقة:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "×",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("معدل الإطارات المصدر:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

img = cv2.imread("assets/samples/lena.jpg")
base = cv2.resize(img, (320, 240))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("demo.mp4", fourcc, 30.0, (320, 240))
print("فُتح الكاتب؟", writer.isOpened())

for i in range(90):
    frame = base.copy()
    x = 10 + i * 3
    cv2.rectangle(frame, (x, 10), (x + 30, 40), (0, 255, 0), -1)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture("demo.mp4")
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
cap.release()

print("عدد الإطارات المقروءة:", count)
print("حجم الملف: %.1f KB" % (os.path.getsize("demo.mp4") / 1024))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import cv2
import os

cap = cv2.VideoCapture("demo.mp4")
os.makedirs("frames", exist_ok=True)
idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % 10 == 0:
        cv2.imwrite("frames/frame_%03d.png" % idx, frame)
        saved += 1
    idx += 1
cap.release()
print("حُفظ", saved, "إطاراً من أصل", idx)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import time
import cv2

img = cv2.imread("assets/samples/lena.jpg")
probe = cv2.resize(img, (320, 240))
probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

def timeit(fn, n=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000

ops = [
    ("تنعيم 7×7", lambda: cv2.GaussianBlur(probe, (7, 7), 0)),
    ("تحويل رمادي", lambda: cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)),
    ("حواف Canny", lambda: cv2.Canny(probe_gray, 100, 200)),
    ("ترميز JPEG", lambda: cv2.imencode(".jpg", probe, [cv2.IMWRITE_JPEG_QUALITY, 90])),
]
for name, fn in ops:
    ms = timeit(fn)
    print("%-12s %5.2f ms/إطار | %5.0f إطار/ثانية" % (name, ms, 1000 / ms))

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/opencv-5-ai-vision/notebooks/ch-03-ch03-io-video.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
last = time.time()
count = 0
fps = 0.0
# داخل حلقة الفيديو:
count += 1
now = time.time()
if now - last >= 1.0:
    fps = count / (now - last)
    count, last = 0, now
