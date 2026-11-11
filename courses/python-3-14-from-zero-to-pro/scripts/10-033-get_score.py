#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-033-get_score.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78
