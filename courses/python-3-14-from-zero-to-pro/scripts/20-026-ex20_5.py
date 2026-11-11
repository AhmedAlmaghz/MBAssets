#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-026-ex20_5.py
"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # صف واحد، عمودان

axes[0].plot(months, sales, marker="o")
axes[0].set_title("خطي")

axes[1].bar(months, sales, color="orange")
axes[1].set_title("شريطي")

plt.tight_layout()
plt.show()
