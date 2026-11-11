#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-033-students.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
students = pd.DataFrame({
    "الاسم": ["أحمد", "سارة", "خالد", "منى", "يوسف"],
    "الرياضيات": [85, 92, 78, 95, 88],
    "العلوم": [90, 85, 82, 91, 79],
})
print(students["الرياضيات"].mean())
print(students["العلوم"].mean())

# 2
numbers = np.random.randint(1, 101, 10)
print(numbers)
print("الأكبر:", numbers.max())
print("الأصغر:", numbers.min())
print("المتوسط:", numbers.mean())

# 3
days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء"]
hours = [2, 3, 1.5, 4, 2.5]
plt.bar(days, hours, color="purple")
plt.title("ساعات الدراسة")
plt.show()
