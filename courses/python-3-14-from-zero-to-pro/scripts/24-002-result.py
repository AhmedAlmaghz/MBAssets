#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-002-result.py
"""

# ❌ سطر طويل جدًا، يصعب قراءته دفعة واحدة
result = calculate_total_price(unit_price, quantity, discount_percentage, tax_rate, shipping_cost)

# ✅ مقسّم بشكل مقروء
result = calculate_total_price(
    unit_price,
    quantity,
    discount_percentage,
    tax_rate,
    shipping_cost,
)
