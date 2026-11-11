#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-005-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة
