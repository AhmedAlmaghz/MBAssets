#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-005-convert.py
"""

# src/currency_converter/converter.py
"""وحدة تحويل العملات."""
from __future__ import annotations

EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
}

class UnsupportedCurrencyError(Exception):
    """تُرفع عندما تكون العملة المطلوبة غير مدعومة."""

def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """
    يحوّل مبلغًا من عملة إلى أخرى بناءً على أسعار الصرف الثابتة.

    Args:
        amount: المبلغ المراد تحويله.
        from_currency: رمز العملة الأصلية (مثل "USD").
        to_currency: رمز العملة الهدف (مثل "SAR").

    Returns:
        المبلغ المحوّل بالعملة الهدف.

    Raises:
        UnsupportedCurrencyError: إن كانت إحدى العملتين غير مدعومة.
    """
    if from_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {from_currency}")
    if to_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {to_currency}")

    amount_in_usd = amount / EXCHANGE_RATES[from_currency]
    return round(amount_in_usd * EXCHANGE_RATES[to_currency], 2)
