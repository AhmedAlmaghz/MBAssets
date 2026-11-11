#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-021-func_any.py
"""

def func_any(x: Any) -> None:
    x.anything_at_all()  # لا تحذير من mypy حتى لو الدالة غير موجودة!

def func_object(x: object) -> None:
    x.anything_at_all()  # mypy يُصدر خطأً! object لا يملك هذه الدالة
