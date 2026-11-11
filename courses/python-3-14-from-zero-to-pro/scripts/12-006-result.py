#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-006-result.py
"""

# بـ map/filter
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# بـ List Comprehension - الأسلوب الـPythonic الموصى به
result = [x ** 2 for x in numbers if x % 2 == 0]
