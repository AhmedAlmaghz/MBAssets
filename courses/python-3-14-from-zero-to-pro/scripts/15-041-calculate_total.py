#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-041-calculate_total.py
"""

# ✅ جيد: التركيز على توقيع الدالة (المعاملات والإرجاع)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)  # mypy يستنتج أن subtotal من نوع float
    return subtotal * (1 + tax_rate)
