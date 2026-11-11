#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.6: العوامل المنطقية: `and`, `or`, `not`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-010-ex6_6.py
"""

is_weekend: bool = False
is_holiday: bool = True

if is_weekend or is_holiday:
    print("يوم إجازة!")
