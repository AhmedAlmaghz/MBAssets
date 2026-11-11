#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-017-parse_age.py
"""

def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")
