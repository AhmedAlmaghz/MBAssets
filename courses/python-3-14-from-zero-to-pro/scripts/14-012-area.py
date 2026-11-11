#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.4: الفئات المجردة (Abstract Base Classes)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-012-area.py
"""

from abc import ABC, abstractmethod

class Shape(ABC):  # ترث من ABC (Abstract Base Class)
    @abstractmethod
    def area(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    def describe(self) -> str:
        """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
        return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# shape = Shape()  
# 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

circle = Circle(5)
print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

# square = Square(5)  
# 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter
