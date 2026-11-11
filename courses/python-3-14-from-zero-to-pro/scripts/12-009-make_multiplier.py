#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-009-make_multiplier.py
"""

def make_multiplier(factor: int):
    """تُرجع دالة تضرب أي رقم بـ factor."""
    def multiplier(number: int) -> int:
        return number * factor  # factor "مُغلَقة" داخل هذه الدالة
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
