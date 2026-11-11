#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-016-calculate_circle_area.py
"""

# ❌ تكرار -- نفس المنطق مكرر 3 مرات
def calculate_circle_area(radius):
    return 3.14159 * radius ** 2

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

def calculate_sphere_volume(radius):
    return (4/3) * 3.14159 * radius ** 3

# ✅ استخراج الثابت المشترك
import math  # استخدام math.pi بدلاً من كتابة الرقم يدويًا في كل مكان

def calculate_circle_area(radius: float) -> float:
    return math.pi * radius ** 2

def calculate_circle_circumference(radius: float) -> float:
    return 2 * math.pi * radius
