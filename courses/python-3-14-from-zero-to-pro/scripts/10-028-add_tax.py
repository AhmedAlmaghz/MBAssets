#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-028-add_tax.py
"""

# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount
