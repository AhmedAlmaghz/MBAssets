#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-032-__init__.py
"""

# تمرين 3
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"مستطيل {self.width}x{self.height}, المساحة={self.area()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return self.area() == other.area()

r1 = Rectangle(4, 5)
r2 = Rectangle(2, 10)
print(r1)              # مستطيل 4x5, المساحة=20
print(r1 == r2)         # True -- نفس المساحة رغم اختلاف الأبعاد!

# تمرين 4
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5/9

print(TemperatureConverter.celsius_to_fahrenheit(100))  # 212.0
print(TemperatureConverter.fahrenheit_to_celsius(32))   # 0.0
