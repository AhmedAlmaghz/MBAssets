#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث: المتغيرات - أول لبنة
القسم 3.6: Python لغة ديناميكية التنميط (Dynamically Typed)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 03-018-age.py
"""

age = 25         # الآن age من نوع int
print(type(age))  # <class 'int'>

age = "خمسة وعشرون"  # ✅ مسموح تمامًا في Python!
print(type(age))      # <class 'str'>
