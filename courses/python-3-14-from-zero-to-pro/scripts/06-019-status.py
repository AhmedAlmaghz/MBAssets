#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.8: الجُمل الشرطية المُدمجة: Ternary Expression

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-019-status.py
"""

age: int = 20

# الطريقة التقليدية
if age >= 18:
    status = "بالغ"
else:
    status = "قاصر"

# الطريقة المختصرة (Ternary / Conditional Expression)
status = "بالغ" if age >= 18 else "قاصر"
