#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التحولات الهندسية: التدوير، التشوه، وإعادة الاستمثال
OpenCV 5 and AI Computer Vision — Geometric

Book slug : opencv-5-ai-vision
Chapter   : ch-07-ch07-geometric (lesson 7)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-07-ch07-geometric.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التحولات الهندسية: التدوير، التشوه، وإعادة الاستمثال
OpenCV 5 and AI Computer Vision — Geometric

Book slug : opencv-5-ai-vision
Chapter   : ch-07-ch07-geometric (lesson 7)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-07-ch07-geometric.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التحولات الهندسية: التدوير، التشوه، وإعادة الاستمثال
OpenCV 5 and AI Computer Vision — Geometric

Book slug : opencv-5-ai-vision
Chapter   : ch-07-ch07-geometric (lesson 7)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
h, w = img.shape[:2]

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 42, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
def rotate_bounds(w, h, angle_deg):
    """حساب المصفوفة والحدود الجديدة بحيث تفلت كل المحتويات دون اقتطاع"""
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)
    return M, (nw, nh)

M, (nw, nh) = rotate_bounds(w, h, 42)
rotated_full = cv2.warpAffine(img, M, (nw, nh))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
half   = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
fixed  = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)  # DbSet للشبكة العصبية

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
pts_src = np.float32([[141, 131], [480, 159], [493, 630], [64, 601]])
pts_dst = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (300, 400))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
h, w = img.shape[:2]
gx, gy = np.meshgrid(np.arange(w), np.arange(h))

# موجة جماهيرية التأثير عربية على المجموعات
amplitude = 10
map_x = (gx + amplitude * np.sin(gy / 20.0)).astype(np.float32)
map_y = (gy + amplitude * np.sin(gx / 25.0)).astype(np.float32)

wavy = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
h, w = img.shape[:2]

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 42, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def rotate_bounds(w, h, angle_deg):
    """حساب المصفوفة والحدود الجديدة بحيث تفلت كل المحتويات دون اقتطاع"""
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)
    return M, (nw, nh)

M, (nw, nh) = rotate_bounds(w, h, 42)
rotated_full = cv2.warpAffine(img, M, (nw, nh))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
half   = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
fixed  = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)  # DbSet للشبكة العصبية

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
pts_src = np.float32([[141, 131], [480, 159], [493, 630], [64, 601]])
pts_dst = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (300, 400))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
h, w = img.shape[:2]
gx, gy = np.meshgrid(np.arange(w), np.arange(h))

# موجة جماهيرية التأثير عربية على المجموعات
amplitude = 10
map_x = (gx + amplitude * np.sin(gy / 20.0)).astype(np.float32)
map_y = (gy + amplitude * np.sin(gx / 25.0)).astype(np.float32)

wavy = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التحولات الهندسية: التدوير، التشوه، وإعادة الاستمثال
OpenCV 5 and AI Computer Vision — Geometric

Book slug : opencv-5-ai-vision
Chapter   : ch-07-ch07-geometric (lesson 7)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
h, w = img.shape[:2]

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 42, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
def rotate_bounds(w, h, angle_deg):
    """حساب المصفوفة والحدود الجديدة بحيث تفلت كل المحتويات دون اقتطاع"""
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)
    return M, (nw, nh)

M, (nw, nh) = rotate_bounds(w, h, 42)
rotated_full = cv2.warpAffine(img, M, (nw, nh))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
half   = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
fixed  = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)  # DbSet للشبكة العصبية

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
pts_src = np.float32([[141, 131], [480, 159], [493, 630], [64, 601]])
pts_dst = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (300, 400))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch07_geometric.md
# ────────────────────────────────────────────────────────────────────────
h, w = img.shape[:2]
gx, gy = np.meshgrid(np.arange(w), np.arange(h))

# موجة جماهيرية التأثير عربية على المجموعات
amplitude = 10
map_x = (gx + amplitude * np.sin(gy / 20.0)).astype(np.float32)
map_y = (gy + amplitude * np.sin(gx / 25.0)).astype(np.float32)

wavy = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
h, w = img.shape[:2]

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 42, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rotate_bounds(w, h, angle_deg):
    """حساب المصفوفة والحدود الجديدة بحيث تفلت كل المحتويات دون اقتطاع"""
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)
    return M, (nw, nh)

M, (nw, nh) = rotate_bounds(w, h, 42)
rotated_full = cv2.warpAffine(img, M, (nw, nh))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
half   = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
fixed  = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)  # DbSet للشبكة العصبية

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
pts_src = np.float32([[141, 131], [480, 159], [493, 630], [64, 601]])
pts_dst = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (300, 400))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-07-ch07-geometric.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
h, w = img.shape[:2]
gx, gy = np.meshgrid(np.arange(w), np.arange(h))

# موجة جماهيرية التأثير عربية على المجموعات
amplitude = 10
map_x = (gx + amplitude * np.sin(gy / 20.0)).astype(np.float32)
map_y = (gy + amplitude * np.sin(gx / 25.0)).astype(np.float32)

wavy = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
