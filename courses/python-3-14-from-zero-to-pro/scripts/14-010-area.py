#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-010-area.py
"""

class Shape:
    def area(self) -> float:
        raise NotImplementedError("يجب تخصيص هذه الدالة في الفئة الفرعية")

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# هنا يكمن سحر تعدد الأشكال:
shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Triangle(3, 8)]

for shape in shapes:
    print(f"{type(shape).__name__}: المساحة = {shape.area():.2f}")
