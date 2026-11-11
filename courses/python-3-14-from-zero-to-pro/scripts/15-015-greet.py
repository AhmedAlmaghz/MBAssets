#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-015-greet.py
"""

# معامل اختياري (له قيمة افتراضية) - لا علاقة له بـ Optional type
def greet(name: str = "ضيف") -> str:
    return f"مرحبًا، {name}"

# Optional type - القيمة قد تكون None
def find(key: str) -> str | None:
    ...
