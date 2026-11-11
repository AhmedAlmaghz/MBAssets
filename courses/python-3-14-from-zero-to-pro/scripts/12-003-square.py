#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-003-square.py
"""

# ❌ ممارسة سيئة - استخدم def بدلاً منها
square = lambda x: x ** 2

# ✅ أفضل
def square(x: int) -> int:
    return x ** 2
