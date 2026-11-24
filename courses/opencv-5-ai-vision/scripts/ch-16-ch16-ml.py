#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — تعلم الآلة الكلاسيكي: cv2.ml من KNN إلى SVM
OpenCV 5 and AI Computer Vision — Ml

Book slug : opencv-5-ai-vision
Chapter   : ch-16-ch16-ml (lesson 16)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-16-ch16-ml.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — تعلم الآلة الكلاسيكي: cv2.ml من KNN إلى SVM
OpenCV 5 and AI Computer Vision — Ml

Book slug : opencv-5-ai-vision
Chapter   : ch-16-ch16-ml (lesson 16)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch16_ml.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# بيانات تدريب: سمتان (2 أبعاد) وثلاثة صنوف
data = np.float32(np.random.randint(0, 60, (25, 2)))
labels = np.random.randint(0, 3, (25, 1)).astype(np.int32)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch16_ml.md
# ────────────────────────────────────────────────────────────────────────
knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

new_sample = np.float32([[35, 22]])            # عينة مجهولة
res, results, neighbours, dist = knn.findNearest(new_sample, k=5)
print("الصنف المتوقع:", int(res))

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch16_ml.md
# ────────────────────────────────────────────────────────────────────────
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)       # نواة RBF للبيانات غير الخطية
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.5)                        # عقوبة الخطأ
svm.setGamma(0.5)                    # عرض «تأثير» النواة

svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred = svm.predict(test_data)[1]     # الصنف المتوقع لكل عينة

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch16_ml.md
# ────────────────────────────────────────────────────────────────────────
dt = cv2.ml.DTrees_create()
dt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred_tree = dt.predict(test_data)[1]

rt = cv2.ml.RTrees_create()          # غابة عشوائية: 50 شجرة افتراضيًا
rt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred_rf  = rt.predict(test_data)[1]

# ────────────────────────────────────────────────────────────────────────
# [005] content/chapters/ch16_ml.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# بيانات تجريبية: سمات مستخرجة من الكونتورات في خط الفرز
features = np.float32([
    [1200, 130, 0.91],   # دائرة
    [1250, 122, 0.93],   # دائرة
    [1400, 128, 0.89],   # دائرة
    [2200, 210, 0.61],   # مستطيل
    [2400, 218, 0.58],   # مستطيل
    [2150, 202, 0.63],   # مستطيل
])
labels = np.int32([0, 0, 0, 1, 1, 1])      # 0=دائرة، 1=مستطيل

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# منتج جديد: مساحة 1500، محيط 140، دائرية 0.87
incoming = np.float32([[1500, 140, 0.87]])
verdict = int(svm.predict(incoming)[1].ravel()[0])
print("التصنيف:", "دائرة" if verdict == 0 else "مستطيل")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-16-ch16-ml.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# بيانات تدريب: سمتان (2 أبعاد) وثلاثة صنوف
data = np.float32(np.random.randint(0, 60, (25, 2)))
labels = np.random.randint(0, 3, (25, 1)).astype(np.int32)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-16-ch16-ml.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

new_sample = np.float32([[35, 22]])            # عينة مجهولة
res, results, neighbours, dist = knn.findNearest(new_sample, k=5)
print("الصنف المتوقع:", int(res))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-16-ch16-ml.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)       # نواة RBF للبيانات غير الخطية
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.5)                        # عقوبة الخطأ
svm.setGamma(0.5)                    # عرض «تأثير» النواة

svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred = svm.predict(test_data)[1]     # الصنف المتوقع لكل عينة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-16-ch16-ml.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
dt = cv2.ml.DTrees_create()
dt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred_tree = dt.predict(test_data)[1]

rt = cv2.ml.RTrees_create()          # غابة عشوائية: 50 شجرة افتراضيًا
rt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
pred_rf  = rt.predict(test_data)[1]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/opencv-5-ai-vision/notebooks/ch-16-ch16-ml.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# بيانات تجريبية: سمات مستخرجة من الكونتورات في خط الفرز
features = np.float32([
    [1200, 130, 0.91],   # دائرة
    [1250, 122, 0.93],   # دائرة
    [1400, 128, 0.89],   # دائرة
    [2200, 210, 0.61],   # مستطيل
    [2400, 218, 0.58],   # مستطيل
    [2150, 202, 0.63],   # مستطيل
])
labels = np.int32([0, 0, 0, 1, 1, 1])      # 0=دائرة، 1=مستطيل

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# منتج جديد: مساحة 1500، محيط 140، دائرية 0.87
incoming = np.float32([[1500, 140, 0.87]])
verdict = int(svm.predict(incoming)[1].ravel()[0])
print("التصنيف:", "دائرة" if verdict == 0 else "مستطيل")
