#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.5: شروط متعددة: `if...elif...else`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-006-grade.py
"""

score: int = 75

if score >= 90:
    grade: str = "ممتاز"
elif score >= 80:
    grade = "جيد جدًا"
elif score >= 70:
    grade = "جيد"
elif score >= 60:
    grade = "مقبول"
else:
    grade = "راسب"

print(f"تقديرك: {grade}")  # تقديرك: جيد
