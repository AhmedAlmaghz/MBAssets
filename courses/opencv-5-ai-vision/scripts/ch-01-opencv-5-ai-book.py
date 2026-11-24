#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — البيئة والأساس: ثورة OpenCV 5 في يد المهندس
OpenCV 5 and AI Computer Vision — Environment and First Program

Book slug : opencv-5-ai-vision
Chapter   : ch-01-opencv-5-ai-book (lesson 1)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-01-opencv-5-ai-book.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — البيئة والأساس: ثورة OpenCV 5 في يد المهندس
OpenCV 5 and AI Computer Vision — Environment and First Program

Book slug : opencv-5-ai-vision
Chapter   : ch-01-opencv-5-ai-book (lesson 1)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-01-opencv-5-ai-book.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — البيئة والأساس: ثورة OpenCV 5 في يد المهندس
OpenCV 5 and AI Computer Vision — Environment and First Program

Book slug : opencv-5-ai-vision
Chapter   : ch-01-opencv-5-ai-book (lesson 1)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# مصفوفة 5×5 من قيم 8 بت — أصغر «صورة» يمكن تحليلها!
img = np.array(
    [[10,  20,  30,  40,  50],
     [15,  25,  35,  45,  55],
     [30,  40,  50,  60,  70],
     [45,  55,  65,  75,  85],
     [60,  70,  80,  90,  100]], dtype=np.uint8)

print(img.shape, img.dtype, img.min(), img.max())
# (5, 5) uint8 10 100

# ────────────────────────────────────────────────────────────────────────
# [002] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
print(cv2.__version__)           # 5.0.0
print(cv2.cuda.getCUDADeviceCount())  # 0 إذا لم تكن هناك بطاقة (وبيئتنا تعمل بدون GPU)

# ────────────────────────────────────────────────────────────────────────
# [003] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) اصنع صورة (أحمر نقي) داخل OpenCV
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, :, 2] = 255        # القناة الثالثة في OpenCV هي الأحمر (BGR!)

# 2) تحيل إلى RGB لنعرضها كورتية بحفظ الرياضة
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('مربع أحمر — والقناة الفاصلة له 2 (في BGR)')
plt.axis('off')
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [004] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("مسارات/مخطوطة/صورة.jpg")   # None إذا فشل!
assert img is not None, "فشل القراءة: تحقق من المسار/الترميز/.TextUtils"

# ────────────────────────────────────────────────────────────────────────
# [005] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"لم أتمكن من قراءة الصورة: {path}")
    return img

# ────────────────────────────────────────────────────────────────────────
# [006] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

cap = cv2.VideoCapture(0)         # 0 = الكاميرا الأولى في النظام
if not cap.isOpened():
    raise RuntimeError("لم أتعرف على الكاميرا")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Frame 7", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # حرف q للخروج
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# مصفوفة 5×5 من قيم 8 بت — أصغر «صورة» يمكن تحليلها!
img = np.array(
    [[10,  20,  30,  40,  50],
     [15,  25,  35,  45,  55],
     [30,  40,  50,  60,  70],
     [45,  55,  65,  75,  85],
     [60,  70,  80,  90,  100]], dtype=np.uint8)

print(img.shape, img.dtype, img.min(), img.max())
# (5, 5) uint8 10 100

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
print(cv2.__version__)           # 5.0.0
print(cv2.cuda.getCUDADeviceCount())  # 0 إذا لم تكن هناك بطاقة (وبيئتنا تعمل بدون GPU)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) اصنع صورة (أحمر نقي) داخل OpenCV
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, :, 2] = 255        # القناة الثالثة في OpenCV هي الأحمر (BGR!)

# 2) تحيل إلى RGB لنعرضها كورتية بحفظ الرياضة
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('مربع أحمر — والقناة الفاصلة له 2 (في BGR)')
plt.axis('off')
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("مسارات/مخطوطة/صورة.jpg")   # None إذا فشل!
assert img is not None, "فشل القراءة: تحقق من المسار/الترميز/.TextUtils"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"لم أتمكن من قراءة الصورة: {path}")
    return img

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

cap = cv2.VideoCapture(0)         # 0 = الكاميرا الأولى في النظام
if not cap.isOpened():
    raise RuntimeError("لم أتعرف على الكاميرا")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Frame 7", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # حرف q للخروج
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — البيئة والأساس: ثورة OpenCV 5 في يد المهندس
OpenCV 5 and AI Computer Vision — Environment and First Program

Book slug : opencv-5-ai-vision
Chapter   : ch-01-opencv-5-ai-book (lesson 1)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# مصفوفة 5×5 من قيم 8 بت — أصغر «صورة» يمكن تحليلها!
img = np.array(
    [[10,  20,  30,  40,  50],
     [15,  25,  35,  45,  55],
     [30,  40,  50,  60,  70],
     [45,  55,  65,  75,  85],
     [60,  70,  80,  90,  100]], dtype=np.uint8)

print(img.shape, img.dtype, img.min(), img.max())
# (5, 5) uint8 10 100

# ────────────────────────────────────────────────────────────────────────
# [002] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
print(cv2.__version__)           # 5.0.0
print(cv2.cuda.getCUDADeviceCount())  # 0 إذا لم تكن هناك بطاقة (وبيئتنا تعمل بدون GPU)

# ────────────────────────────────────────────────────────────────────────
# [003] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) اصنع صورة (أحمر نقي) داخل OpenCV
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, :, 2] = 255        # القناة الثالثة في OpenCV هي الأحمر (BGR!)

# 2) تحيل إلى RGB لنعرضها كورتية بحفظ الرياضة
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('مربع أحمر — والقناة الفاصلة له 2 (في BGR)')
plt.axis('off')
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [004] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("مسارات/مخطوطة/صورة.jpg")   # None إذا فشل!
assert img is not None, "فشل القراءة: تحقق من المسار/الترميز/.TextUtils"

# ────────────────────────────────────────────────────────────────────────
# [005] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"لم أتمكن من قراءة الصورة: {path}")
    return img

# ────────────────────────────────────────────────────────────────────────
# [006] output/OpenCV_5_AI_Book.html
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

cap = cv2.VideoCapture(0)         # 0 = الكاميرا الأولى في النظام
if not cap.isOpened():
    raise RuntimeError("لم أتعرف على الكاميرا")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Frame 7", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # حرف q للخروج
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# مصفوفة 5×5 من قيم 8 بت — أصغر «صورة» يمكن تحليلها!
img = np.array(
    [[10,  20,  30,  40,  50],
     [15,  25,  35,  45,  55],
     [30,  40,  50,  60,  70],
     [45,  55,  65,  75,  85],
     [60,  70,  80,  90,  100]], dtype=np.uint8)

print(img.shape, img.dtype, img.min(), img.max())
# (5, 5) uint8 10 100

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import cv2
print(cv2.__version__)           # 5.0.0
print(cv2.cuda.getCUDADeviceCount())  # 0 إذا لم تكن هناك بطاقة (وبيئتنا تعمل بدون GPU)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) اصنع صورة (أحمر نقي) داخل OpenCV
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, :, 2] = 255        # القناة الثالثة في OpenCV هي الأحمر (BGR!)

# 2) تحيل إلى RGB لنعرضها كورتية بحفظ الرياضة
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('مربع أحمر — والقناة الفاصلة له 2 (في BGR)')
plt.axis('off')
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("مسارات/مخطوطة/صورة.jpg")   # None إذا فشل!
assert img is not None, "فشل القراءة: تحقق من المسار/الترميز/.TextUtils"

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"لم أتمكن من قراءة الصورة: {path}")
    return img

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-01-opencv-5-ai-book.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import time

cap = cv2.VideoCapture(0)         # 0 = الكاميرا الأولى في النظام
if not cap.isOpened():
    raise RuntimeError("لم أتعرف على الكاميرا")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Frame 7", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # حرف q للخروج
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
