#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.7: النصوص غير قابلة للتعديل (Immutable) - مفهوم حاسم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-019-ex4_7.py
"""

word: str = "مرحبا"
word[0] = "M"  # ❌ TypeError: 'str' object does not support item assignment
