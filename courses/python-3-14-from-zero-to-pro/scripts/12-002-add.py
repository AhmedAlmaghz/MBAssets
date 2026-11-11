#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-002-add.py
"""

# lambda مع عدة معاملات
add = lambda a, b: a + b
print(add(3, 4))  # 7

# lambda مع Ternary Expression
is_even = lambda n: "زوجي" if n % 2 == 0 else "فردي"
print(is_even(7))  # فردي

# lambda بدون معاملات
greet = lambda: "مرحباً!"
print(greet())  # مرحباً!
