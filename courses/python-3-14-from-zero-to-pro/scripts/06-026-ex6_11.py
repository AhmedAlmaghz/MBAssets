#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.11: الشروط المتداخلة (Nested Conditions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-026-ex6_11.py
"""

age: int = 25
has_id: bool = True

if age >= 18:
    if has_id:
        print("يمكنك الدخول")
    else:
        print("تحتاج إثبات هوية")
else:
    print("عمرك غير كافٍ")
