#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-039-point.py
"""

from typing import TypedDict

class Point(TypedDict):
    x: int
    y: int

p: Point = {"x": 1}  # ❌ ناقص حقل y
