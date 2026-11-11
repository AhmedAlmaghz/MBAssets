#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-006-ex9_2.py
"""

# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)
