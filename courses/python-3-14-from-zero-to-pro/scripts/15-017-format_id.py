#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-017-format_id.py
"""

def format_id(value: int | str) -> str:
    """يحوّل معرّفًا (رقمًا أو نصًا) إلى نص منسّق."""
    return f"ID-{value}"

print(format_id(42))      # "ID-42"
print(format_id("A001"))  # "ID-A001"
