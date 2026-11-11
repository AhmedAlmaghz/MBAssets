#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-008-scores.py
"""

scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")
