#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-016-student.py
"""

@dataclass
class Student:
    grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed
