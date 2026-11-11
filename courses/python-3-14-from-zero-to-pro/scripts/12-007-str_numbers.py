#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-007-str_numbers.py
"""

# استخدام map مع دالة موجودة - مقبول تماماً وشائع
str_numbers = ["1", "2", "3"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3]
