#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الصورة الرقمية من الداخل: البكسلات والقنوات والعمق
OpenCV 5 and AI Computer Vision — Tmedia

Book slug : opencv-5-ai-vision
Chapter   : ch-02-ch02-tmedia (lesson 2)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-02-ch02-tmedia.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الصورة الرقمية من الداخل: البكسلات والقنوات والعمق
OpenCV 5 and AI Computer Vision — Tmedia

Book slug : opencv-5-ai-vision
Chapter   : ch-02-ch02-tmedia (lesson 2)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-02-ch02-tmedia.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الصورة الرقمية من الداخل: البكسلات والقنوات والعمق
OpenCV 5 and AI Computer Vision — Tmedia

Book slug : opencv-5-ai-vision
Chapter   : ch-02-ch02-tmedia (lesson 2)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

print("الشكل:", img.shape, "| النوع:", img.dtype)
print("البكسل [100, 50]:", img[100, 50].tolist())
print("أول خمسة بكسلات في الصف 0:")
print(img[0, 0:5].tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("BGR عند (0, 0):", int(b[0, 0]), int(g[0, 0]), int(r[0, 0]))
print("RGB عند (0, 0):", img_rgb[0, 0].tolist())
print("متوسطات B/G/R: %.1f / %.1f / %.1f"
      # % (float(b.mean()), float(g.mean()), float(r.mean())))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("الرمادي:", gray.shape, gray.dtype, "| المتوسط: %.2f" % float(gray.mean()))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

grad = np.linspace(0, 255, 256, dtype=np.uint8)
q3 = (np.round(grad / 255 * 7) / 7 * 255).astype(np.uint8)
print("قيم فريدة قبل التكميم:", len(np.unique(grad)))
print("قيم فريدة بعده:", len(np.unique(q3)))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
test = img.copy()
view = test[100:150, 100:150]

before = int(test[120, 120, 0])
view[:] = 0
after = int(test[120, 120, 0])

print("هل يتشاركان الذاكرة؟", np.shares_memory(view, test))
print("قبل التعديل:", before, "| بعده:", after)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300].copy()
print("نسخة مستقلة:", np.shares_memory(roi, img))
print("شكلها:", roi.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import cv2

a = np.array([250, 254, 255], dtype=np.uint8)
print("جمع NumPy:", (a + np.uint8(5)).tolist())
print("جمع OpenCV:", cv2.add(a.reshape(1, 3), np.full((1, 3), 5, np.uint8)).tolist())

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300]          # [y1:y2, x1:x2]
print("شكل ROI:", roi.shape)
print("متوسط القناة الزرقاء: %.2f" % float(roi[:, :, 0].mean()))

roi_copy = roi.copy()
roi_copy[:] = 0                       # تعديل النسخة فقط
print("قناة B عند (120, 120) في الأصل:", int(img[120, 120, 0]))

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
cv2.imwrite("roi.png", roi)
print("حُفظت منطقة الاهتمام في: roi.png")

# ────────────────────────────────────────────────────────────────────────
# [010] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, ch) in zip(axes, [("الأزرق", b), ("الأخضر", g), ("الأحمر", r)]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("channels_view.png", dpi=120)
print("أبعاد القنوات:", b.shape, g.shape, r.shape)

# ────────────────────────────────────────────────────────────────────────
# [011] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
blend = cv2.addWeighted(img, 0.35, gray_bgr, 0.65, 0)

print("اللون: المتوسط %.2f | الانحراف %.2f" % (float(img.mean()), float(img.std())))
print("المزيج: المتوسط %.2f | الانحراف %.2f" % (float(blend.mean()), float(blend.std())))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

print("الشكل:", img.shape, "| النوع:", img.dtype)
print("البكسل [100, 50]:", img[100, 50].tolist())
print("أول خمسة بكسلات في الصف 0:")
print(img[0, 0:5].tolist())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("BGR عند (0, 0):", int(b[0, 0]), int(g[0, 0]), int(r[0, 0]))
print("RGB عند (0, 0):", img_rgb[0, 0].tolist())
print("متوسطات B/G/R: %.1f / %.1f / %.1f"
      # % (float(b.mean()), float(g.mean()), float(r.mean())))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("الرمادي:", gray.shape, gray.dtype, "| المتوسط: %.2f" % float(gray.mean()))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

grad = np.linspace(0, 255, 256, dtype=np.uint8)
q3 = (np.round(grad / 255 * 7) / 7 * 255).astype(np.uint8)
print("قيم فريدة قبل التكميم:", len(np.unique(grad)))
print("قيم فريدة بعده:", len(np.unique(q3)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
test = img.copy()
view = test[100:150, 100:150]

before = int(test[120, 120, 0])
view[:] = 0
after = int(test[120, 120, 0])

print("هل يتشاركان الذاكرة؟", np.shares_memory(view, test))
print("قبل التعديل:", before, "| بعده:", after)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300].copy()
print("نسخة مستقلة:", np.shares_memory(roi, img))
print("شكلها:", roi.shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import cv2

a = np.array([250, 254, 255], dtype=np.uint8)
print("جمع NumPy:", (a + np.uint8(5)).tolist())
print("جمع OpenCV:", cv2.add(a.reshape(1, 3), np.full((1, 3), 5, np.uint8)).tolist())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300]          # [y1:y2, x1:x2]
print("شكل ROI:", roi.shape)
print("متوسط القناة الزرقاء: %.2f" % float(roi[:, :, 0].mean()))

roi_copy = roi.copy()
roi_copy[:] = 0                       # تعديل النسخة فقط
print("قناة B عند (120, 120) في الأصل:", int(img[120, 120, 0]))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
cv2.imwrite("roi.png", roi)
print("حُفظت منطقة الاهتمام في: roi.png")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, ch) in zip(axes, [("الأزرق", b), ("الأخضر", g), ("الأحمر", r)]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("channels_view.png", dpi=120)
print("أبعاد القنوات:", b.shape, g.shape, r.shape)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
blend = cv2.addWeighted(img, 0.35, gray_bgr, 0.65, 0)

print("اللون: المتوسط %.2f | الانحراف %.2f" % (float(img.mean()), float(img.std())))
print("المزيج: المتوسط %.2f | الانحراف %.2f" % (float(blend.mean()), float(blend.std())))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الصورة الرقمية من الداخل: البكسلات والقنوات والعمق
OpenCV 5 and AI Computer Vision — Tmedia

Book slug : opencv-5-ai-vision
Chapter   : ch-02-ch02-tmedia (lesson 2)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

print("الشكل:", img.shape, "| النوع:", img.dtype)
print("البكسل [100, 50]:", img[100, 50].tolist())
print("أول خمسة بكسلات في الصف 0:")
print(img[0, 0:5].tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("BGR عند (0, 0):", int(b[0, 0]), int(g[0, 0]), int(r[0, 0]))
print("RGB عند (0, 0):", img_rgb[0, 0].tolist())
print("متوسطات B/G/R: %.1f / %.1f / %.1f"
      # % (float(b.mean()), float(g.mean()), float(r.mean())))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("الرمادي:", gray.shape, gray.dtype, "| المتوسط: %.2f" % float(gray.mean()))

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

grad = np.linspace(0, 255, 256, dtype=np.uint8)
q3 = (np.round(grad / 255 * 7) / 7 * 255).astype(np.uint8)
print("قيم فريدة قبل التكميم:", len(np.unique(grad)))
print("قيم فريدة بعده:", len(np.unique(q3)))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
test = img.copy()
view = test[100:150, 100:150]

before = int(test[120, 120, 0])
view[:] = 0
after = int(test[120, 120, 0])

print("هل يتشاركان الذاكرة؟", np.shares_memory(view, test))
print("قبل التعديل:", before, "| بعده:", after)

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300].copy()
print("نسخة مستقلة:", np.shares_memory(roi, img))
print("شكلها:", roi.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import cv2

a = np.array([250, 254, 255], dtype=np.uint8)
print("جمع NumPy:", (a + np.uint8(5)).tolist())
print("جمع OpenCV:", cv2.add(a.reshape(1, 3), np.full((1, 3), 5, np.uint8)).tolist())

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300]          # [y1:y2, x1:x2]
print("شكل ROI:", roi.shape)
print("متوسط القناة الزرقاء: %.2f" % float(roi[:, :, 0].mean()))

roi_copy = roi.copy()
roi_copy[:] = 0                       # تعديل النسخة فقط
print("قناة B عند (120, 120) في الأصل:", int(img[120, 120, 0]))

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
cv2.imwrite("roi.png", roi)
print("حُفظت منطقة الاهتمام في: roi.png")

# ────────────────────────────────────────────────────────────────────────
# [010] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, ch) in zip(axes, [("الأزرق", b), ("الأخضر", g), ("الأحمر", r)]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("channels_view.png", dpi=120)
print("أبعاد القنوات:", b.shape, g.shape, r.shape)

# ────────────────────────────────────────────────────────────────────────
# [011] content/chapters/ch02_tmedia.md
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
blend = cv2.addWeighted(img, 0.35, gray_bgr, 0.65, 0)

print("اللون: المتوسط %.2f | الانحراف %.2f" % (float(img.mean()), float(img.std())))
print("المزيج: المتوسط %.2f | الانحراف %.2f" % (float(blend.mean()), float(blend.std())))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
assert img is not None, "فشلت قراءة الصورة: تحقق من المسار"

print("الشكل:", img.shape, "| النوع:", img.dtype)
print("البكسل [100, 50]:", img[100, 50].tolist())
print("أول خمسة بكسلات في الصف 0:")
print(img[0, 0:5].tolist())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("BGR عند (0, 0):", int(b[0, 0]), int(g[0, 0]), int(r[0, 0]))
print("RGB عند (0, 0):", img_rgb[0, 0].tolist())
print("متوسطات B/G/R: %.1f / %.1f / %.1f"
      # % (float(b.mean()), float(g.mean()), float(r.mean())))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("الرمادي:", gray.shape, gray.dtype, "| المتوسط: %.2f" % float(gray.mean()))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

grad = np.linspace(0, 255, 256, dtype=np.uint8)
q3 = (np.round(grad / 255 * 7) / 7 * 255).astype(np.uint8)
print("قيم فريدة قبل التكميم:", len(np.unique(grad)))
print("قيم فريدة بعده:", len(np.unique(q3)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
test = img.copy()
view = test[100:150, 100:150]

before = int(test[120, 120, 0])
view[:] = 0
after = int(test[120, 120, 0])

print("هل يتشاركان الذاكرة؟", np.shares_memory(view, test))
print("قبل التعديل:", before, "| بعده:", after)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300].copy()
print("نسخة مستقلة:", np.shares_memory(roi, img))
print("شكلها:", roi.shape)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import cv2

a = np.array([250, 254, 255], dtype=np.uint8)
print("جمع NumPy:", (a + np.uint8(5)).tolist())
print("جمع OpenCV:", cv2.add(a.reshape(1, 3), np.full((1, 3), 5, np.uint8)).tolist())

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img = cv2.imread("assets/samples/lena.jpg")
roi = img[100:250, 150:300]          # [y1:y2, x1:x2]
print("شكل ROI:", roi.shape)
print("متوسط القناة الزرقاء: %.2f" % float(roi[:, :, 0].mean()))

roi_copy = roi.copy()
roi_copy[:] = 0                       # تعديل النسخة فقط
print("قناة B عند (120, 120) في الأصل:", int(img[120, 120, 0]))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
cv2.imwrite("roi.png", roi)
print("حُفظت منطقة الاهتمام في: roi.png")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/samples/lena.jpg")
b, g, r = cv2.split(img)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, ch) in zip(axes, [("الأزرق", b), ("الأخضر", g), ("الأحمر", r)]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("channels_view.png", dpi=120)
print("أبعاد القنوات:", b.shape, g.shape, r.shape)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/opencv-5-ai-vision/notebooks/ch-02-ch02-tmedia.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import cv2

img = cv2.imread("assets/samples/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
blend = cv2.addWeighted(img, 0.35, gray_bgr, 0.65, 0)

print("اللون: المتوسط %.2f | الانحراف %.2f" % (float(img.mean()), float(img.std())))
print("المزيج: المتوسط %.2f | الانحراف %.2f" % (float(blend.mean()), float(blend.std())))
