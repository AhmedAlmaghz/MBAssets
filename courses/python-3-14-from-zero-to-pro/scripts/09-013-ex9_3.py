#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-013-ex9_3.py
"""

fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)  
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!
