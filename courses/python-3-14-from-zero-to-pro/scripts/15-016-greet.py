#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-016-greet.py
"""

def greet(name: str | None = None) -> str:
    if name is None:
        name = "ضيف"
    return f"مرحبًا، {name}"
