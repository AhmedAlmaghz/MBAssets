#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-005-prices.py
"""

prices = np.array([100, 200, 300, 400])

# عملية على كل عنصر دفعة واحدة
with_tax = prices * 1.15
print(with_tax)  # [115. 230. 345. 460.]

# مقارنة تُرجع مصفوفة من Boolean
expensive = prices > 250
print(expensive)  # [False False  True  True]

# استخدام هذه المصفوفة Boolean للتصفية! (Boolean Indexing)
print(prices[expensive])  # [300 400]

# أو في سطر واحد
print(prices[prices > 250])  # [300 400]
