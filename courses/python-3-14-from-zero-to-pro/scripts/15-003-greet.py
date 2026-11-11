#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-003-greet.py
"""

age: int = 25
name: str = "محمد"

def greet(name: str) -> str:
    return f"مرحبًا، {name}!"
