#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-013-my_decorator.py
"""

def my_decorator(func):
    """decorator بسيط يطبع رسالة قبل وبعد تنفيذ الدالة."""
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()  # استدعاء الدالة الأصلية
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

def say_hello():
    print("مرحباً!")

# التطبيق اليدوي (بدون @)
decorated_hello = my_decorator(say_hello)
decorated_hello()
