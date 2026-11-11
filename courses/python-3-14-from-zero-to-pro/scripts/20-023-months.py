#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-023-months.py
"""

import matplotlib.pyplot as plt

months = ["يناير", "فبراير", "مارس", "أبريل", "مايو"]
sales = [1200, 1900, 1500, 2200, 2800]

plt.plot(months, sales)
plt.title("المبيعات الشهرية")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.show()  # يفتح نافذة لعرض الرسم
