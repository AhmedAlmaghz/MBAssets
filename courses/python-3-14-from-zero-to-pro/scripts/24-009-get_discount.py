#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-009-get_discount.py
"""

# ❌ سيء: تداخل عميق (Nested Ifs) صعب التتبع
def get_discount(user):
    if user is not None:
        if user.is_active:
            if user.membership == "premium":
                return 0.2
            else:
                return 0.1
        else:
            return 0
    else:
        return 0

# ✅ جيد: Guard Clauses -- خروج مبكر، أقل تداخلًا
def get_discount(user) -> float:
    if user is None:
        return 0
    if not user.is_active:
        return 0
    if user.membership == "premium":
        return 0.2
    return 0.1
