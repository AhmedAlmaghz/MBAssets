#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-011-__init__.py
"""

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width    # attribute
        self.height = height  # attribute

    def area(self) -> float:  # method
        return self.width * self.height

    def perimeter(self) -> float:  # method
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect.width)        # 5 - الوصول لـ attribute (بدون أقواس)
print(rect.area())       # 15.0 - استدعاء method (مع أقواس)
