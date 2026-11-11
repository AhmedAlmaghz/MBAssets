#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-014-point.py
"""

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)          # Point(x=3, y=4)  -- __repr__ تلقائياً!
print(p1 == p2)     # True  -- __eq__ تلقائياً!
