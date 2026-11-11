#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-020-process_anything.py
"""

from typing import Any

def process_anything(data: Any) -> Any:
    return data
