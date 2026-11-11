#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-017-full_example.py
"""

def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")
