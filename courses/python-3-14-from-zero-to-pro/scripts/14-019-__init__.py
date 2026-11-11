#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-019-__init__.py
"""

class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, value: float) -> float:
        return value * self.factor

double = Multiplier(2)
print(double(5))   # 10 -- الكائن نفسه يُستدعى كدالة!
print(double(10))  # 20
