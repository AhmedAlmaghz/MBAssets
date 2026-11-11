#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-026-increment.py
"""

# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2
