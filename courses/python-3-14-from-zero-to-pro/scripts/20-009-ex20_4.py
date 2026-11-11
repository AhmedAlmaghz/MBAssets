#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-009-ex20_4.py
"""

# من CSV
df = pd.read_csv("sales.csv")

# من Excel (يتطلب: uv add openpyxl)
df = pd.read_excel("sales.xlsx", sheet_name="2024")

# من JSON
df = pd.read_json("sales.json")

# تحديد الترميز إن كانت البيانات عربية
df = pd.read_csv("sales.csv", encoding="utf-8")
