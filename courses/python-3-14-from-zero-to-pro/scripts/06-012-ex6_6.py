#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.6: العوامل المنطقية: `and`, `or`, `not`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-012-ex6_6.py
"""

age: int = 25
has_ticket: bool = True
is_vip: bool = False

if (age >= 18 and has_ticket) or is_vip:
    print("مسموح بالدخول")
