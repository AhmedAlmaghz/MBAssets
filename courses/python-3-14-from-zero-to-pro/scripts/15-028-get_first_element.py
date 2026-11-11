#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-028-get_first_element.py
"""

def get_first_element(items: list) -> Any:
    return items[0]
