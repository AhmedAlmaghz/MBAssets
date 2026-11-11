#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-027-ex8_8.py
"""

numbers: list[int] = range(1, 6)
labels: list[str] = ["زوجي" if n % 2 == 0 else "فردي" for n in numbers]
print(labels)  # ['فردي', 'زوجي', 'فردي', 'زوجي', 'فردي']
