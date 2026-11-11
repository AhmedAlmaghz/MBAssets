#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.11: الشروط المتداخلة (Nested Conditions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-027-ex6_11.py
"""

# بدلاً من التداخل العميق
if age >= 18:
    if has_id:
        if not is_banned:
            print("مسموح")

# الأفضل: دمج الشروط
if age >= 18 and has_id and not is_banned:
    print("مسموح")
