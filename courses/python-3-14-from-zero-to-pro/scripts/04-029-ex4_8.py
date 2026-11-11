#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.8: أهم Methods النصوص (الأكثر استخدامًا عمليًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-029-ex4_8.py
"""

age_input: str = input("أدخل عمرك: ")

if age_input.isdigit():
    age: int = int(age_input)
    print(f"عمرك {age} سنة")
else:
    print("خطأ: يجب إدخال رقم فقط!")
