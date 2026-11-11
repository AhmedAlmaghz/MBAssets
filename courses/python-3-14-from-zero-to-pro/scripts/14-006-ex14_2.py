#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-006-ex14_2.py
"""

# ❌ لا يأخذ الوراثة في الحسبان
if type(dog) == Animal:  # False حتى لو dog هو كلب (نوع فرعي من Animal)!
    pass

# ✅ يأخذ الوراثة في الحسبان
if isinstance(dog, Animal):  # True
    pass
