#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-004-add.py
"""

# متغيرات
age: int = 25
price: float = 19.99
name: str = "سارة"
is_active: bool = True

# دوال
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"مرحبًا، {name}")
