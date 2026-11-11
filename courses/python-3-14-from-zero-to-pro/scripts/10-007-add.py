#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-007-add.py
"""

# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9
