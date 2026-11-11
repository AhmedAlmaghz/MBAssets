#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-025-ex20_5.py
"""

plt.figure(figsize=(10, 6))  # حجم الرسم (عرض, ارتفاع) بالبوصة

plt.plot(months, sales, marker="o", linestyle="--", color="green", linewidth=2, label="2024")
plt.title("المبيعات الشهرية", fontsize=16, fontweight="bold")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.legend()        # إظهار الـ label الموجود أعلاه
plt.grid(True, alpha=0.3)   # شبكة خفيفة في الخلفية
plt.tight_layout()  # ترتيب تلقائي لمنع تداخل العناصر

plt.savefig("sales_chart.png", dpi=300)  # حفظ كصورة (بدلاً من/بالإضافة إلى show)
plt.show()
