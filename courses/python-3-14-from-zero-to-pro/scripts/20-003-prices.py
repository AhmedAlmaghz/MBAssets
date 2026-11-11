#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-003-prices.py
"""

import numpy as np

prices = np.array([10, 20, 30])
discounts = np.array([1, 2, 3])

result = prices - discounts
print(result)  # [9 18 27]
print(type(result))  # <class 'numpy.ndarray'>
