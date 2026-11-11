#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-019-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # <__main__.Point object at 0x7f8a1c0b3d90>  -- غير مفيد!
