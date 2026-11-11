#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-024-ex8_8.py
"""

names: list[str] = ["محمد", "سارة", "أحمد"]
uppercase_names: list[str] = [name.upper() for name in names]
print(uppercase_names)
