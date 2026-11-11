#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-006-sales.py
"""

sales = np.array([120, 340, 210, 450, 90, 380])

print(sales.sum())     # 1590 -- المجموع
print(sales.mean())    # 265.0 -- المتوسط
print(sales.max())     # 450 -- الأكبر
print(sales.min())     # 90 -- الأصغر
print(sales.std())     # الانحراف المعياري
print(sales.argmax())  # 3 -- فهرس العنصر الأكبر
print(np.median(sales))  # الوسيط
