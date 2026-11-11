#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-004-students.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

# مع sorted() - الاستخدام الأشهر لـ lambda
sorted_by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in sorted_by_score:
    print(s["name"], s["score"])

# مع min() و max()
youngest_scorer = min(students, key=lambda s: s["score"])
print(youngest_scorer)
