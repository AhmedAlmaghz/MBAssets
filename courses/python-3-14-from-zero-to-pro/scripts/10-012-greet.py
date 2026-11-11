#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-012-greet.py
"""

def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!
