#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-023-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}
