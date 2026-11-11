#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-018-format_id.py
"""

from typing import Union

def format_id(value: Union[int, str]) -> str:
    return f"ID-{value}"
