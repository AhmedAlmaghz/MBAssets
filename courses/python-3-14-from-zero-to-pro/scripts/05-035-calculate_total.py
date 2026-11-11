#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-035-calculate_total.py
"""

def calculate_total(price: float, quantity: int) -> float:
    return round(price * quantity, 2)
