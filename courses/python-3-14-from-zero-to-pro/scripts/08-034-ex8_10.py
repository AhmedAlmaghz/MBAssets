#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.10: القوائم والحلقات: دمج كل ما تعلمناه

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-034-ex8_10.py
"""

for student, score in zip(students, scores):
    status: str = "ناجح ✅" if score >= 60 else "راسب ❌"
    print(f"{student}: {score} - {status}")
