#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-002-add.py
"""

# src/calculator.py
def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def divide(a: float, b: float) -> float:
    """قسمة رقمين، مع رفع استثناء عند القسمة على صفر."""
    if b == 0:
        raise ValueError("لا يمكن القسمة على صفر")
    return a / b
