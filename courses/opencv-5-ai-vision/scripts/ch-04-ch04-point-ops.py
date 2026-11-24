#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العمليات على البكسل: التسوية الديناميكية والحسابات النقطية
OpenCV 5 and AI Computer Vision — Point Ops

Book slug : opencv-5-ai-vision
Chapter   : ch-04-ch04-point-ops (lesson 4)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-04-ch04-point-ops.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العمليات على البكسل: التسوية الديناميكية والحسابات النقطية
OpenCV 5 and AI Computer Vision — Point Ops

Book slug : opencv-5-ai-vision
Chapter   : ch-04-ch04-point-ops (lesson 4)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-04-ch04-point-ops.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العمليات على البكسل: التسوية الديناميكية والحسابات النقطية
OpenCV 5 and AI Computer Vision — Point Ops

Book slug : opencv-5-ai-vision
Chapter   : ch-04-ch04-point-ops (lesson 4)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

print("شكل الصورة:", img.shape, "| النوع:", img.dtype)
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(brighter.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[10]], np.uint8)
print("النتيجة:", cv2.convertScaleAbs(probe, alpha=1.0, beta=-40)[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
manual = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).round().astype(np.uint8)

diff = cv2.absdiff(brighter, manual)
print("أقصى فرق بين الطريقتين:", int(diff.max()))
print("متوسط الفرق: %.4f" % float(diff.mean()))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gamma = 0.55
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
gamma_img = cv2.LUT(img, lut)

print("عدد مدخلات الجدول:", len(lut))
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(gamma_img.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(gray)
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask)
print("شكل القناع:", mask.shape, "| القيم الفريدة:", np.unique(mask))
print("نسبة المساحة المختارة: %.1f%%" % (100 * np.count_nonzero(mask) / mask.size))
print("شكل الناتج:", masked.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)  # محاكاة تباين منخفض
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)

def normalize_img(a, lo=0, hi=255):
    f = a.astype(np.float32)
    mn, mx = f.min(), f.max()
    if mx == mn:
        return np.zeros_like(a)
    return ((f - mn) / (mx - mn) * (hi - lo) + lo).astype(np.uint8)

full = normalize_img(low_gray)
print("قبل: min=%d max=%d std=%.2f" % (low_gray.min(), low_gray.max(), low_gray.std()))
print("بعد : min=%d max=%d std=%.2f" % (full.min(), full.max(), full.std()))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
full = cv2.normalize(low_gray, None, 0, 255, cv2.NORM_MINMAX)
print("أدنى قيمة:", int(full.min()), "| أعلى قيمة:", int(full.max()))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)

print("الانحراف المعياري — الأصل: %.1f | equalizeHist: %.1f | CLAHE: %.1f"
      # % (float(gray.std()), float(eq.std()), float(cl.std())))

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
dark = cv2.convertScaleAbs(img, alpha=0.45, beta=0)  # محاكاة تعرّض منخفض

gamma_lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
step1 = cv2.LUT(dark, gamma_lut)
step2 = cv2.convertScaleAbs(step1, alpha=1.15, beta=10)

h, w = step2.shape[:2]
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)
focused = cv2.bitwise_and(step2, step2, mask=mask)

print("متوسط الأصلية: %.2f" % float(img.mean()))
print("متوسط المظلمة: %.2f" % float(dark.mean()))
print("متوسط بعد جاما: %.2f" % float(step1.mean()))
print("متوسط بعد التباين: %.2f" % float(step2.mean()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

print("شكل الصورة:", img.shape, "| النوع:", img.dtype)
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(brighter.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[10]], np.uint8)
print("النتيجة:", cv2.convertScaleAbs(probe, alpha=1.0, beta=-40)[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
manual = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).round().astype(np.uint8)

diff = cv2.absdiff(brighter, manual)
print("أقصى فرق بين الطريقتين:", int(diff.max()))
print("متوسط الفرق: %.4f" % float(diff.mean()))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gamma = 0.55
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
gamma_img = cv2.LUT(img, lut)

print("عدد مدخلات الجدول:", len(lut))
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(gamma_img.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(gray)
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask)
print("شكل القناع:", mask.shape, "| القيم الفريدة:", np.unique(mask))
print("نسبة المساحة المختارة: %.1f%%" % (100 * np.count_nonzero(mask) / mask.size))
print("شكل الناتج:", masked.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)  # محاكاة تباين منخفض
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)

def normalize_img(a, lo=0, hi=255):
    f = a.astype(np.float32)
    mn, mx = f.min(), f.max()
    if mx == mn:
        return np.zeros_like(a)
    return ((f - mn) / (mx - mn) * (hi - lo) + lo).astype(np.uint8)

full = normalize_img(low_gray)
print("قبل: min=%d max=%d std=%.2f" % (low_gray.min(), low_gray.max(), low_gray.std()))
print("بعد : min=%d max=%d std=%.2f" % (full.min(), full.max(), full.std()))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
full = cv2.normalize(low_gray, None, 0, 255, cv2.NORM_MINMAX)
print("أدنى قيمة:", int(full.min()), "| أعلى قيمة:", int(full.max()))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)

print("الانحراف المعياري — الأصل: %.1f | equalizeHist: %.1f | CLAHE: %.1f"
      # % (float(gray.std()), float(eq.std()), float(cl.std())))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
dark = cv2.convertScaleAbs(img, alpha=0.45, beta=0)  # محاكاة تعرّض منخفض

gamma_lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
step1 = cv2.LUT(dark, gamma_lut)
step2 = cv2.convertScaleAbs(step1, alpha=1.15, beta=10)

h, w = step2.shape[:2]
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)
focused = cv2.bitwise_and(step2, step2, mask=mask)

print("متوسط الأصلية: %.2f" % float(img.mean()))
print("متوسط المظلمة: %.2f" % float(dark.mean()))
print("متوسط بعد جاما: %.2f" % float(step1.mean()))
print("متوسط بعد التباين: %.2f" % float(step2.mean()))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — العمليات على البكسل: التسوية الديناميكية والحسابات النقطية
OpenCV 5 and AI Computer Vision — Point Ops

Book slug : opencv-5-ai-vision
Chapter   : ch-04-ch04-point-ops (lesson 4)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

print("شكل الصورة:", img.shape, "| النوع:", img.dtype)
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(brighter.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[10]], np.uint8)
print("النتيجة:", cv2.convertScaleAbs(probe, alpha=1.0, beta=-40)[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
manual = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).round().astype(np.uint8)

diff = cv2.absdiff(brighter, manual)
print("أقصى فرق بين الطريقتين:", int(diff.max()))
print("متوسط الفرق: %.4f" % float(diff.mean()))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gamma = 0.55
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
gamma_img = cv2.LUT(img, lut)

print("عدد مدخلات الجدول:", len(lut))
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(gamma_img.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(gray)
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask)
print("شكل القناع:", mask.shape, "| القيم الفريدة:", np.unique(mask))
print("نسبة المساحة المختارة: %.1f%%" % (100 * np.count_nonzero(mask) / mask.size))
print("شكل الناتج:", masked.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)  # محاكاة تباين منخفض
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)

def normalize_img(a, lo=0, hi=255):
    f = a.astype(np.float32)
    mn, mx = f.min(), f.max()
    if mx == mn:
        return np.zeros_like(a)
    return ((f - mn) / (mx - mn) * (hi - lo) + lo).astype(np.uint8)

full = normalize_img(low_gray)
print("قبل: min=%d max=%d std=%.2f" % (low_gray.min(), low_gray.max(), low_gray.std()))
print("بعد : min=%d max=%d std=%.2f" % (full.min(), full.max(), full.std()))

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
full = cv2.normalize(low_gray, None, 0, 255, cv2.NORM_MINMAX)
print("أدنى قيمة:", int(full.min()), "| أعلى قيمة:", int(full.max()))

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)

print("الانحراف المعياري — الأصل: %.1f | equalizeHist: %.1f | CLAHE: %.1f"
      # % (float(gray.std()), float(eq.std()), float(cl.std())))

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch04_point_ops.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
dark = cv2.convertScaleAbs(img, alpha=0.45, beta=0)  # محاكاة تعرّض منخفض

gamma_lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
step1 = cv2.LUT(dark, gamma_lut)
step2 = cv2.convertScaleAbs(step1, alpha=1.15, beta=10)

h, w = step2.shape[:2]
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)
focused = cv2.bitwise_and(step2, step2, mask=mask)

print("متوسط الأصلية: %.2f" % float(img.mean()))
print("متوسط المظلمة: %.2f" % float(dark.mean()))
print("متوسط بعد جاما: %.2f" % float(step1.mean()))
print("متوسط بعد التباين: %.2f" % float(step2.mean()))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

print("شكل الصورة:", img.shape, "| النوع:", img.dtype)
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(brighter.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
probe = np.array([[10]], np.uint8)
print("النتيجة:", cv2.convertScaleAbs(probe, alpha=1.0, beta=-40)[0, 0])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
alpha, beta = 1.3, 40
brighter = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
manual = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).round().astype(np.uint8)

diff = cv2.absdiff(brighter, manual)
print("أقصى فرق بين الطريقتين:", int(diff.max()))
print("متوسط الفرق: %.4f" % float(diff.mean()))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gamma = 0.55
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
gamma_img = cv2.LUT(img, lut)

print("عدد مدخلات الجدول:", len(lut))
print("متوسط السطوع قبل:", round(float(img.mean()), 2))
print("متوسط السطوع بعد:", round(float(gamma_img.mean()), 2))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(gray)
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask)
print("شكل القناع:", mask.shape, "| القيم الفريدة:", np.unique(mask))
print("نسبة المساحة المختارة: %.1f%%" % (100 * np.count_nonzero(mask) / mask.size))
print("شكل الناتج:", masked.shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)  # محاكاة تباين منخفض
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)

def normalize_img(a, lo=0, hi=255):
    f = a.astype(np.float32)
    mn, mx = f.min(), f.max()
    if mx == mn:
        return np.zeros_like(a)
    return ((f - mn) / (mx - mn) * (hi - lo) + lo).astype(np.uint8)

full = normalize_img(low_gray)
print("قبل: min=%d max=%d std=%.2f" % (low_gray.min(), low_gray.max(), low_gray.std()))
print("بعد : min=%d max=%d std=%.2f" % (full.min(), full.max(), full.std()))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
low = cv2.convertScaleAbs(img, alpha=0.35, beta=70)
low_gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
full = cv2.normalize(low_gray, None, 0, 255, cv2.NORM_MINMAX)
print("أدنى قيمة:", int(full.min()), "| أعلى قيمة:", int(full.max()))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)

print("الانحراف المعياري — الأصل: %.1f | equalizeHist: %.1f | CLAHE: %.1f"
      # % (float(gray.std()), float(eq.std()), float(cl.std())))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/opencv-5-ai-vision/notebooks/ch-04-ch04-point-ops.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
dark = cv2.convertScaleAbs(img, alpha=0.45, beta=0)  # محاكاة تعرّض منخفض

gamma_lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
step1 = cv2.LUT(dark, gamma_lut)
step2 = cv2.convertScaleAbs(step1, alpha=1.15, beta=10)

h, w = step2.shape[:2]
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w // 2, h // 2), 150, 255, -1)
focused = cv2.bitwise_and(step2, step2, mask=mask)

print("متوسط الأصلية: %.2f" % float(img.mean()))
print("متوسط المظلمة: %.2f" % float(dark.mean()))
print("متوسط بعد جاما: %.2f" % float(step1.mean()))
print("متوسط بعد التباين: %.2f" % float(step2.mean()))
