#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث: المتغيرات - أول لبنة
القسم 3.8: تحويل الأنواع (Type Casting)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 03-024-ex3_8.py
"""

age: str = input("كم عمرك؟ ")
age_number: int = int(age)  # التحويل الصريح إلى int
age_next_year: int = age_number + 1
print(age_next_year)  # يعمل بنجاح الآن
