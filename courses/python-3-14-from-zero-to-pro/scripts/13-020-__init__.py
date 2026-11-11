#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-020-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """يُستدعى عند print() أو str() - للعرض البشري المقروء."""
        return f"نقطة({self.x}, {self.y})"

    def __repr__(self) -> str:
        """يُستدعى في REPL أو داخل قوائم - للعرض التقني الدقيق."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(p)       # نقطة(3, 4)  -- يستخدم __str__
print([p])     # [Point(x=3, y=4)]  -- القوائم تستخدم __repr__ للعناصر
print(repr(p)) # Point(x=3, y=4)
