#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-018-summary.py
"""

# عدة إحصائيات دفعة واحدة
summary = df.groupby("المدينة")["الإجمالي"].agg(["sum", "mean", "count", "max"])
print(summary)

# التجميع حسب أكثر من عمود
by_city_product = df.groupby(["المدينة", "المنتج"])["الإجمالي"].sum()
print(by_city_product)

# تحويل النتيجة إلى DataFrame عادي مرة أخرى
result = df.groupby("المدينة")["الإجمالي"].sum().reset_index()
