#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.10: دمج النصوص: عمليات متعددة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-032-full_name.py
"""

first_name: str = "محمد"
last_name: str = "أحمد"

# الطريقة 1: عامل + (Concatenation)
full_name: str = first_name + " " + last_name
print(full_name)  # محمد أحمد

# الطريقة 2: f-string (الأفضل والأكثر مرونة)
full_name = f"{first_name} {last_name}"

# الطريقة 3: تكرار النص بعامل *
separator: str = "-" * 20
print(separator)  # --------------------
