#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-017-is_even.py
"""

# ❌ معقّد دون داعٍ
def is_even(n: int) -> bool:
    return True if n % 2 == 0 else False

# ✅ بسيط ومباشر
def is_even(n: int) -> bool:
    return n % 2 == 0
