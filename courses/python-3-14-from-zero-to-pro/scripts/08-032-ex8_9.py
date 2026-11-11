#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.9: دوال مُدمجة قوية تعمل مع القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-032-ex8_9.py
"""

names: list[str] = ["محمد", "سارة", "أحمد"]
ages: list[int] = [25, 30, 28]

for name, age in zip(names, ages):
    print(f"{name}: {age} سنة")
