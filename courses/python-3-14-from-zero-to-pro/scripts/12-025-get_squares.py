#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-025-get_squares.py
"""

def get_squares(n: int) -> list[int]:
    """تُنشئ قائمة كاملة بكل المربعات - تستهلك ذاكرة كبيرة!"""
    return [i ** 2 for i in range(n)]

# إذا n = 10,000,000، هذا ينشئ قائمة ضخمة بالكامل في الذاكرة فوراً!
squares = get_squares(10_000_000)
