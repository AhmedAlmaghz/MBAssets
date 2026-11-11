#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-019-pivot.py
"""

pivot = df.pivot_table(
    values="الإجمالي",
    index="المدينة",
    columns="المنتج",
    aggfunc="sum",
    fill_value=0,
)
print(pivot)
