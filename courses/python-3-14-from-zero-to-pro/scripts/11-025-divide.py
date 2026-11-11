#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.9: مقدمة مختصرة لـ logging بدلاً من print

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-025-divide.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)
