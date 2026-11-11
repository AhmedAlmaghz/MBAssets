#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-005-print_separator.py
"""

def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None
