#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.10: مقارنة الأرقام: عوامل المقارنة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-027-ex5_10.py
"""

result: float = 0.1 + 0.2
is_close_enough: bool = abs(result - 0.3) < 0.0001
print(is_close_enough)  # True
