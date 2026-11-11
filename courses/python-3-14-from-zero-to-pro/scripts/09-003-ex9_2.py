#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-003-ex9_2.py
"""

# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25
