#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-007-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}
