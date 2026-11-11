#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.9: match-case: الأسلوب الحديث الثوري (Python 3.10+)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-021-ex6_9.py
"""

day: str = "Monday"

match day:
    case "Monday":
        print("بداية الأسبوع")
    case "Friday":
        print("نهاية أسبوع العمل")
    case "Saturday" | "Sunday":  # عامل | يعني "أو" هنا
        print("عطلة نهاية الأسبوع")
    case _:
        print("يوم عادي")
