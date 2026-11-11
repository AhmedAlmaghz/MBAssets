#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.12: Tuples: الأخ «غير القابل للتعديل» للقائمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-036-ex8_12.py
"""

coordinates: tuple[int, int] = (10, 20)
coordinates[0] = 99  # ❌ TypeError: 'tuple' object does not support item assignment
