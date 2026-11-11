#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-004-ex20_3.py
"""

import numpy as np

# من قائمة
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.dtype)    # int64 (نوع البيانات الموحّد)

# مصفوفة ثنائية الأبعاد (مصفوفة من مصفوفات = جدول)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(matrix.shape)  # (2, 3) -- صفّان، 3 أعمدة

# دوال إنشاء جاهزة
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))        # مصفوفة 2x3 من الآحاد
range_arr = np.arange(0, 10, 2)   # [0 2 4 6 8]  (مثل range لكن تُرجع array)
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ] -- 5 قيم متباعدة بالتساوي
random_arr = np.random.rand(3)    # 3 أرقام عشوائية بين 0 و1
