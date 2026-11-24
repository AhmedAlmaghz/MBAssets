#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التدرجات ومرشحات الحواف: من Sobel إلى Canny
OpenCV 5 and AI Computer Vision — Edges

Book slug : opencv-5-ai-vision
Chapter   : ch-06-ch06-edges (lesson 6)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-06-ch06-edges.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التدرجات ومرشحات الحواف: من Sobel إلى Canny
OpenCV 5 and AI Computer Vision — Edges

Book slug : opencv-5-ai-vision
Chapter   : ch-06-ch06-edges (lesson 6)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-06-ch06-edges.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التدرجات ومرشحات الحواف: من Sobel إلى Canny
OpenCV 5 and AI Computer Vision — Edges

Book slug : opencv-5-ai-vision
Chapter   : ch-06-ch06-edges (lesson 6)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/building.jpg", cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # مشتق أفقية
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # مشتق عمودي

magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

angle = cv2.phase(gx, gy, angleInDegrees=True)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
log_edges = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
edges = cv2.Canny(gray, threshold1=90, threshold2=180, apertureSize=3, L2gradient=False)

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

edges_c = auto_canny(gray)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.6)      # قبل الحواف: تنعيم خفيف
    edges = cv2.Canny(blur, 80, 160)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('frame | edges', np.hstack([frame, edges_color]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/building.jpg", cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # مشتق أفقية
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # مشتق عمودي

magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

angle = cv2.phase(gx, gy, angleInDegrees=True)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
log_edges = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
edges = cv2.Canny(gray, threshold1=90, threshold2=180, apertureSize=3, L2gradient=False)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

edges_c = auto_canny(gray)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.6)      # قبل الحواف: تنعيم خفيف
    edges = cv2.Canny(blur, 80, 160)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('frame | edges', np.hstack([frame, edges_color]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التدرجات ومرشحات الحواف: من Sobel إلى Canny
OpenCV 5 and AI Computer Vision — Edges

Book slug : opencv-5-ai-vision
Chapter   : ch-06-ch06-edges (lesson 6)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/building.jpg", cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # مشتق أفقية
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # مشتق عمودي

magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

angle = cv2.phase(gx, gy, angleInDegrees=True)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
log_edges = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
edges = cv2.Canny(gray, threshold1=90, threshold2=180, apertureSize=3, L2gradient=False)

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

edges_c = auto_canny(gray)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch06_edges.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.6)      # قبل الحواف: تنعيم خفيف
    edges = cv2.Canny(blur, 80, 160)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('frame | edges', np.hstack([frame, edges_color]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

gray = cv2.imread("assets/samples/building.jpg", cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # مشتق أفقية
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # مشتق عمودي

magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

angle = cv2.phase(gx, gy, angleInDegrees=True)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
log_edges = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
edges = cv2.Canny(gray, threshold1=90, threshold2=180, apertureSize=3, L2gradient=False)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

edges_c = auto_canny(gray)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-06-ch06-edges.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.6)      # قبل الحواف: تنعيم خفيف
    edges = cv2.Canny(blur, 80, 160)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('frame | edges', np.hstack([frame, edges_color]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
