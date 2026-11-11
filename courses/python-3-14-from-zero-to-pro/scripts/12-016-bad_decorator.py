#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-016-bad_decorator.py
"""

def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
        # لا يوجد return هنا!
    return wrapper

@bad_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # None  💥 بدلاً من 8!
