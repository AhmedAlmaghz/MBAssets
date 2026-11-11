#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.1: لماذا تُعتبر الدوال نقطة تحوّل في رحلتك البرمجية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-001-calculate_circle_area.py
"""

# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")
