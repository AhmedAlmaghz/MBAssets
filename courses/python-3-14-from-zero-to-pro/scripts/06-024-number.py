#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.10: Walrus Operator `:=` - الأداة الذكية (Python 3.8+)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-024-number.py
"""

# بدون walrus operator - تكرار الحساب مرتين
import random
number = random.randint(1, 10)

if number > 5:
    print(f"الرقم {number} أكبر من 5")
