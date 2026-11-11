#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.8: List Comprehension: الأسلوب الأنيق والـ"Pythonic"

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-030-result.py
"""

# ❌ list comprehension معقدة جدًا - صعبة القراءة
result = [x**2 if x % 2 == 0 else x**3 for x in range(20) if x > 5 if x < 15]

# ✅ أوضح كحلقة تقليدية في هذه الحالة المعقدة
result = []
for x in range(20):
    if 5 < x < 15:
        if x % 2 == 0:
            result.append(x ** 2)
        else:
            result.append(x ** 3)
