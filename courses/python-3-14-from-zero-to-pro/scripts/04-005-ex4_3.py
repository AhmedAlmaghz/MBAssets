#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.3: الحروف الخاصة (Escape Characters)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-005-ex4_3.py
"""

# بدون raw string - مشكلة!
path: str = "C:\new_folder"  # \n هنا تُفسَّر كسطر جديد! خطأ غير متوقع
print(path)
# الناتج: C:
# ew_folder  ← خطأ!

# مع raw string - الحل
path: str = r"C:\new_folder"
print(path)
# الناتج: C:\new_folder  ← صحيح تمامًا
