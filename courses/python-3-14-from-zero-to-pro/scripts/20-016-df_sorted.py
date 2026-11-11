#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-016-df_sorted.py
"""

df_sorted = df.sort_values("السعر")                    # تصاعدي
df_sorted = df.sort_values("السعر", ascending=False)    # تنازلي
df_sorted = df.sort_values(["المدينة", "السعر"])         # فرز متعدد المستويات
