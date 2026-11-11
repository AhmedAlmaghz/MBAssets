#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-005-add.py
"""

def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")  # لا يوجد أي خطأ من Python!
print(result)  # "مرحبًا!"
