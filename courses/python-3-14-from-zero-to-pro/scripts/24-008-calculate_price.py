#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-008-calculate_price.py
"""

# ❌ سيء: ما معنى 0.15؟ ولماذا 3؟
def calculate_price(amount):
    if amount > 1000:
        return amount * 0.85
    return amount * 0.95

for attempt in range(3):
    try_connection()

# ✅ جيد: الثوابت المُسمّاة تشرح نفسها
BULK_DISCOUNT_THRESHOLD = 1000
BULK_DISCOUNT_RATE = 0.85
STANDARD_DISCOUNT_RATE = 0.95
MAX_CONNECTION_ATTEMPTS = 3

def calculate_price(amount: float) -> float:
    if amount > BULK_DISCOUNT_THRESHOLD:
        return amount * BULK_DISCOUNT_RATE
    return amount * STANDARD_DISCOUNT_RATE

for attempt in range(MAX_CONNECTION_ATTEMPTS):
    try_connection()
