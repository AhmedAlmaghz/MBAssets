#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-024-categories.py
"""

import matplotlib.pyplot as plt

categories = ["لابتوب", "ماوس", "شاشة", "لوحة مفاتيح"]
values = [45, 120, 30, 80]

# رسم بياني خطي -- مناسب للاتجاهات عبر الزمن
plt.plot(categories, values, marker="o")
plt.show()

# رسم بياني شريطي (Bar Chart) -- مناسب للمقارنة بين فئات
plt.bar(categories, values, color="steelblue")
plt.show()

# رسم دائري (Pie Chart) -- مناسب لإظهار النسب من الكل
plt.pie(values, labels=categories, autopct="%1.1f%%")
plt.show()

# مخطط التشتت (Scatter Plot) -- مناسب لإظهار العلاقة بين متغيرين
prices = [3500, 50, 900, 120]
plt.scatter(prices, values)
plt.xlabel("السعر")
plt.ylabel("الكمية المباعة")
plt.show()

# المدرج التكراري (Histogram) -- مناسب لتوزيع البيانات
import numpy as np
ages = np.random.normal(30, 5, 1000)  # 1000 عمر عشوائي حول متوسط 30
plt.hist(ages, bins=20, color="coral", edgecolor="black")
plt.show()
