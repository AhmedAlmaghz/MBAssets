#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-015-numbers.py
"""

numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!
