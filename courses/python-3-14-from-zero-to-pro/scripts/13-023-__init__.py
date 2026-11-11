#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-023-__init__.py
"""

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # يستدعي v1.__add__(v2) تلقائياً!
print(v3)  # Vector(4, 6)
