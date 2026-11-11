#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-011-summarize_students.py
"""

def summarize_students(
    grades: dict[str, list[int]]
) -> dict[str, float]:
    """
    يستقبل قاموسًا بأسماء الطلاب ودرجاتهم،
    ويُرجع قاموسًا بأسماء الطلاب ومتوسط درجاتهم.
    """
    averages: dict[str, float] = {}
    for student, scores in grades.items():
        averages[student] = sum(scores) / len(scores)
    return averages

student_grades: dict[str, list[int]] = {
    "محمد": [85, 90, 78],
    "فاطمة": [92, 88, 95],
}

result: dict[str, float] = summarize_students(student_grades)
print(result)
# {'محمد': 84.33333333333333, 'فاطمة': 91.66666666666667}
