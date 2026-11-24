#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الخصائص ومعايرة الكاميرا: من الاختصاصات إلى العالم ثلاثي الأبعاد
OpenCV 5 and AI Computer Vision — Features

Book slug : opencv-5-ai-vision
Chapter   : ch-15-ch15-features (lesson 15)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-15-ch15-features.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — الخصائص ومعايرة الكاميرا: من الاختصاصات إلى العالم ثلاثي الأبعاد
OpenCV 5 and AI Computer Vision — Features

Book slug : opencv-5-ai-vision
Chapter   : ch-15-ch15-features (lesson 15)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img1 = cv2.imread("assets/samples/box.png", 0)
img2 = cv2.imread("assets/samples/box_in_scene.png", 0)

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print("زوايا عنصر1:", len(kp1), "| زوايا عنصر2:", len(kp2))

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
# 1) مطابقة قسرية (Brute Force): تجربة كل النقاط
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)   # أقرب المسافات أفضل

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                              flags=cv2.DRAW_RICH_KEYPOINTS)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)   # رقم تحقق الغير

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1_f, des2_f, k=2)   # لسلّم نسبة Lowe

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
print("مطابقات جيدة:", len(good), "من", len(matches))

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M2, mask = cv2.findHomography(src_pts, dst_pts,
                              cv2.RANSAC, 5.0)
inliers = mask.ravel().sum()
print("نقاط داخلية:", inliers, "/" , len(good))

# ────────────────────────────────────────────────────────────────────────
# [006] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
h, w = img1.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M2)
img_out = cv2.polylines(img2.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

# ────────────────────────────────────────────────────────────────────────
# [007] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
# 1) مُعدّ الشطرج والأمثلة المركزية
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 25.0  # 25 mm لكل مربع

objpoints = []    # نقاط العالم ثلاثي الأبعاد
imgpoints = []    # نقطتًا في الصورة (2D)

for fname in ["left01.jpg", "right01.jpg"]:
    img = cv2.imread(f"assets/samples/{fname}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

# 2) معايرة الكاميرا
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("مصفوفة الكاميرا:\n", K)
print("معاملات التشويه:\n", dist.ravel())

# ────────────────────────────────────────────────────────────────────────
# [008] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
undist = cv2.undistort(img, K, dist)

# ────────────────────────────────────────────────────────────────────────
# [009] content/chapters/ch15_features.md
# ────────────────────────────────────────────────────────────────────────
stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=64, blockSize=9)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

img1 = cv2.imread("assets/samples/box.png", 0)
img2 = cv2.imread("assets/samples/box_in_scene.png", 0)

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print("زوايا عنصر1:", len(kp1), "| زوايا عنصر2:", len(kp2))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) مطابقة قسرية (Brute Force): تجربة كل النقاط
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)   # أقرب المسافات أفضل

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                              flags=cv2.DRAW_RICH_KEYPOINTS)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)   # رقم تحقق الغير

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1_f, des2_f, k=2)   # لسلّم نسبة Lowe

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
print("مطابقات جيدة:", len(good), "من", len(matches))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M2, mask = cv2.findHomography(src_pts, dst_pts,
                              cv2.RANSAC, 5.0)
inliers = mask.ravel().sum()
print("نقاط داخلية:", inliers, "/" , len(good))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
h, w = img1.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M2)
img_out = cv2.polylines(img2.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 1) مُعدّ الشطرج والأمثلة المركزية
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 25.0  # 25 mm لكل مربع

objpoints = []    # نقاط العالم ثلاثي الأبعاد
imgpoints = []    # نقطتًا في الصورة (2D)

for fname in ["left01.jpg", "right01.jpg"]:
    img = cv2.imread(f"assets/samples/{fname}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

# 2) معايرة الكاميرا
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("مصفوفة الكاميرا:\n", K)
print("معاملات التشويه:\n", dist.ravel())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
undist = cv2.undistort(img, K, dist)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/opencv-5-ai-vision/notebooks/ch-15-ch15-features.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=64, blockSize=9)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
