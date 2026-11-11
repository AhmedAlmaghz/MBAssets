#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-010-increment.py
"""

class Counter:
    def increment():  # ❌ نسيان self!
        pass

c = Counter()
c.increment()  
# 💥 TypeError: increment() takes 0 positional arguments but 1 was given
