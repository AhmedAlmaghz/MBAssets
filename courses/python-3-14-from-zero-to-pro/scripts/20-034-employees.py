#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-034-employees.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# 4
employees = pd.DataFrame({
    "اسم": ["علي", "فاطمة", "سعد", "ليلى", "عمر"],
    "القسم": ["مبيعات", "تقنية", "مبيعات", "تقنية", "مالية"],
    "الراتب": [8000, 12000, 8500, 13000, 9500],
})
avg_salary = employees.groupby("القسم")["الراتب"].mean()
print(avg_salary)
avg_salary.plot(kind="bar", color="teal")
plt.title("متوسط الراتب حسب القسم")
plt.ylabel("الراتب")
plt.tight_layout()
plt.show()

# 5
data = pd.DataFrame({"العمر": [25, 30, None, 45, 28, None, 90]})
# الوسيط أفضل من المتوسط هنا لأن القيمة 90 (قيمة متطرفة/Outlier)
# تؤثر بشدة على المتوسط لكن تأثيرها على الوسيط محدود جدًا
data["العمر"] = data["العمر"].fillna(data["العمر"].median())
print(data)
