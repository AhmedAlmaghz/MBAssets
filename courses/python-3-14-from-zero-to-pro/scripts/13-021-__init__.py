#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-021-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(5, 6)

print(p1 == p2)  # True  -- بدون __eq__ كانت ستكون False (مقارنة مراجع الذاكرة)!
print(p1 == p3)  # False
