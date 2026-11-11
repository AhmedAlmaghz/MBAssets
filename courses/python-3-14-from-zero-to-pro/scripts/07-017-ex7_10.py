#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.10: `break` و `continue` في الحلقات المتداخلة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-017-ex7_10.py
"""

for i in range(3):
    for j in range(3):
        if j == 1:
            break  # يُخرج فقط من الحلقة الداخلية!
        print(f"i={i}, j={j}")
