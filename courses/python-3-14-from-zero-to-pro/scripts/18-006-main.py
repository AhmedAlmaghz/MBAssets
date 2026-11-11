#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-006-main.py
"""

# src/currency_converter/main.py
"""نقطة الدخول الرئيسية للبرنامج."""
from currency_converter.converter import convert, UnsupportedCurrencyError

def main() -> None:
    try:
        result = convert(100, "USD", "SAR")
        print(f"100 USD = {result} SAR")
    except UnsupportedCurrencyError as e:
        print(f"خطأ: {e}")

if __name__ == "__main__":
    main()
