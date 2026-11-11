#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-014-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)
