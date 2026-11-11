#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-018-immutable_point.py
"""

from dataclasses import dataclass

@dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(3, 4)
# p.x = 10  
# 💥 FrozenInstanceError: cannot assign to field 'x'

@dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
class Version:
    major: int
    minor: int

v1 = Version(1, 5)
v2 = Version(2, 0)
print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف
