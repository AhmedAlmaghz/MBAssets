#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.10: القوائم والحلقات: دمج كل ما تعلمناه

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-033-ex8_10.py
"""

students: list[str] = ["محمد", "سارة", "أحمد", "فاطمة"]
scores: list[int] = [85, 92, 78, 95]

for index, student in enumerate(students):
    score: int = scores[index]
    status: str = "ناجح ✅" if score >= 60 else "راسب ❌"
    print(f"{student}: {score} - {status}")
