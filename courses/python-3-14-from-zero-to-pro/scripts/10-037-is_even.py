#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-037-is_even.py
"""

# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0
