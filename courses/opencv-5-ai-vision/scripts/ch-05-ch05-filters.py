#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الترشيح والالتفاف: المرشحات، التنعيم، وحدود النظام
OpenCV 5 and AI Computer Vision — Filters

Book slug : opencv-5-ai-vision
Chapter   : ch-05-ch05-filters (lesson 5)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-05-ch05-filters.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الترشيح والالتفاف: المرشحات، التنعيم، وحدود النظام
OpenCV 5 and AI Computer Vision — Filters

Book slug : opencv-5-ai-vision
Chapter   : ch-05-ch05-filters (lesson 5)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-05-ch05-filters.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الترشيح والالتفاف: المرشحات، التنعيم، وحدود النظام
OpenCV 5 and AI Computer Vision — Filters

Book slug : opencv-5-ai-vision
Chapter   : ch-05-ch05-filters (lesson 5)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")

box = cv2.blur(img, (7, 7))                      # box blur (بـ المتوسط)
gauss = cv2.GaussianBlur(img, (7, 7), sigmaX=1.8) # gauss (۷ mediante بلا ترانزيت!!)
median = cv2.medianBlur(img, 5)                  # median (Philippines ksize فردي فقط)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                   [0, -1,  0]], dtype=np.float32)   # شاربر بسلعة كلاسيكية

sharp = cv2.filter2D(img, ddepth=-1, kernel=kernel)   # -1 = نفس عمق الصورة

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)

# 1) اضف ضجيج سأم البيان (salt & pepper) كسيناريو
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, 0, 35)                     # ضجيج gauss وليس عند المضلع
noisy = cv2.add(img, noise)

# 2) أجرب: بلس + درجة الاسترداد مرورية
gauss_fixed = cv2.GaussianBlur(noisy, (5, 5), 1.2)
median_fixed = cv2.medianBlur(noisy, 5)

# 3) أحدد الأفضل المفيد تصوّرًا: عبر PSNR (نسبة الإشارة إلى الضجيج)
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 10 * np.log10((255**2) / mse) if mse > 0 else float("inf")

print("PSNR gauss:", round(psnr(gauss_fixed, img), 2),
      "| PSNR median:", round(psnr(median_fixed, img), 2))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")

box = cv2.blur(img, (7, 7))                      # box blur (بـ المتوسط)
gauss = cv2.GaussianBlur(img, (7, 7), sigmaX=1.8) # gauss (۷ mediante بلا ترانزيت!!)
median = cv2.medianBlur(img, 5)                  # median (Philippines ksize فردي فقط)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                   [0, -1,  0]], dtype=np.float32)   # شاربر بسلعة كلاسيكية

sharp = cv2.filter2D(img, ddepth=-1, kernel=kernel)   # -1 = نفس عمق الصورة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)

# 1) اضف ضجيج سأم البيان (salt & pepper) كسيناريو
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, 0, 35)                     # ضجيج gauss وليس عند المضلع
noisy = cv2.add(img, noise)

# 2) أجرب: بلس + درجة الاسترداد مرورية
gauss_fixed = cv2.GaussianBlur(noisy, (5, 5), 1.2)
median_fixed = cv2.medianBlur(noisy, 5)

# 3) أحدد الأفضل المفيد تصوّرًا: عبر PSNR (نسبة الإشارة إلى الضجيج)
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 10 * np.log10((255**2) / mse) if mse > 0 else float("inf")

print("PSNR gauss:", round(psnr(gauss_fixed, img), 2),
      "| PSNR median:", round(psnr(median_fixed, img), 2))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الترشيح والالتفاف: المرشحات، التنعيم، وحدود النظام
OpenCV 5 and AI Computer Vision — Filters

Book slug : opencv-5-ai-vision
Chapter   : ch-05-ch05-filters (lesson 5)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")

box = cv2.blur(img, (7, 7))                      # box blur (بـ المتوسط)
gauss = cv2.GaussianBlur(img, (7, 7), sigmaX=1.8) # gauss (۷ mediante بلا ترانزيت!!)
median = cv2.medianBlur(img, 5)                  # median (Philippines ksize فردي فقط)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                   [0, -1,  0]], dtype=np.float32)   # شاربر بسلعة كلاسيكية

sharp = cv2.filter2D(img, ddepth=-1, kernel=kernel)   # -1 = نفس عمق الصورة

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch05_filters.md
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)

# 1) اضف ضجيج سأم البيان (salt & pepper) كسيناريو
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, 0, 35)                     # ضجيج gauss وليس عند المضلع
noisy = cv2.add(img, noise)

# 2) أجرب: بلس + درجة الاسترداد مرورية
gauss_fixed = cv2.GaussianBlur(noisy, (5, 5), 1.2)
median_fixed = cv2.medianBlur(noisy, 5)

# 3) أحدد الأفضل المفيد تصوّرًا: عبر PSNR (نسبة الإشارة إلى الضجيج)
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 10 * np.log10((255**2) / mse) if mse > 0 else float("inf")

print("PSNR gauss:", round(psnr(gauss_fixed, img), 2),
      "| PSNR median:", round(psnr(median_fixed, img), 2))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")

box = cv2.blur(img, (7, 7))                      # box blur (بـ المتوسط)
gauss = cv2.GaussianBlur(img, (7, 7), sigmaX=1.8) # gauss (۷ mediante بلا ترانزيت!!)
median = cv2.medianBlur(img, 5)                  # median (Philippines ksize فردي فقط)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                   [0, -1,  0]], dtype=np.float32)   # شاربر بسلعة كلاسيكية

sharp = cv2.filter2D(img, ddepth=-1, kernel=kernel)   # -1 = نفس عمق الصورة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-05-ch05-filters.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
img = cv2.imread("assets/samples/board.jpg", cv2.IMREAD_GRAYSCALE)

# 1) اضف ضجيج سأم البيان (salt & pepper) كسيناريو
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, 0, 35)                     # ضجيج gauss وليس عند المضلع
noisy = cv2.add(img, noise)

# 2) أجرب: بلس + درجة الاسترداد مرورية
gauss_fixed = cv2.GaussianBlur(noisy, (5, 5), 1.2)
median_fixed = cv2.medianBlur(noisy, 5)

# 3) أحدد الأفضل المفيد تصوّرًا: عبر PSNR (نسبة الإشارة إلى الضجيج)
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 10 * np.log10((255**2) / mse) if mse > 0 else float("inf")

print("PSNR gauss:", round(psnr(gauss_fixed, img), 2),
      "| PSNR median:", round(psnr(median_fixed, img), 2))
