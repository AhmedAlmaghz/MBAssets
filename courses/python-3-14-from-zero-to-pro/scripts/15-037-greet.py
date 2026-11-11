#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-037-greet.py
"""

def greet(name: str, greeting: str = "مرحبًا") -> str:
    return f"{greeting}، {name}!"

print(greet())  # ❌ نسينا معاملًا إلزاميًا
