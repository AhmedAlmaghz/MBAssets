#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-009-create_profile.py
"""

def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))
