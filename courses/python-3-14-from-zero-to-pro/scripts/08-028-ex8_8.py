#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-028-ex8_8.py
"""

# shرط Filter (يُكتب في النهاية)
[x for x in range(10) if x % 2 == 0]

# شرط Ternary (يُكتب قبل for مباشرة)
[x if x % 2 == 0 else -x for x in range(10)]
