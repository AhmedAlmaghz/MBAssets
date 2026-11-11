#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-002-ex9_1.py
"""

student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!
