#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.8: الجُمل الشرطية المُدمجة: Ternary Expression

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-020-discount.py
"""

# ✅ استخدام جيد للـ ternary - بسيط وواضح
discount = 0.2 if is_vip else 0.0

# ❌ استخدام سيء - معقد جدًا وغير مقروء كـ ternary
result = "A" if x > 90 else "B" if x > 80 else "C" if x > 70 else "D"
# في هذه الحالة، استخدم if/elif/else التقليدية بدلاً من هذا!
