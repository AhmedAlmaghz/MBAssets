#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-013-__init__.py
"""

# الطريقة التقليدية - كثير من الكود المتكرر لفئة بسيطة تخزن بيانات فقط
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
