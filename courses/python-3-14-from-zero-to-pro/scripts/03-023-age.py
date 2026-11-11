#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث: المتغيرات - أول لبنة
القسم 3.8: تحويل الأنواع (Type Casting)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 03-023-age.py
"""

age = input("كم عمرك؟ ")  # المستخدم يكتب: 25
print(type(age))            # <class 'str'> ← ليس int!

# محاولة استخدامه كرقم مباشرة:
age_next_year = age + 1  # ❌ TypeError: can only concatenate str (not "int") to str
