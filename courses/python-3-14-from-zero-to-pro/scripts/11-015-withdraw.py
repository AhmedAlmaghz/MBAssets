#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-015-withdraw.py
"""

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")
