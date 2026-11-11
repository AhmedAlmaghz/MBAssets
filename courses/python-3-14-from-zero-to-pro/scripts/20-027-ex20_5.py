#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-027-ex20_5.py
"""

import pandas as pd

df = pd.DataFrame({
    "الشهر": ["يناير", "فبراير", "مارس"],
    "المبيعات": [1200, 1900, 1500],
})

df.plot(x="الشهر", y="المبيعات", kind="bar", title="المبيعات الشهرية")
plt.show()

# أو مباشرة من نتيجة groupby
df.groupby("الشهر")["المبيعات"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.show()
