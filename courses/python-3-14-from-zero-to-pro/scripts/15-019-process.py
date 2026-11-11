#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-019-process.py
"""

def process(value: int | str) -> None:
    if isinstance(value, int):
        print(f"رقم مضاعف: {value * 2}")
    else:
        print(f"نص بأحرف كبيرة: {value.upper()}")
