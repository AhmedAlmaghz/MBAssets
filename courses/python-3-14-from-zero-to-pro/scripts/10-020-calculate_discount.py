#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-020-calculate_discount.py
"""

def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد
