#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-014-my_decorator.py
"""

def my_decorator(func):
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

@my_decorator  # هذا يُعادل: say_hello = my_decorator(say_hello)
def say_hello():
    print("مرحباً!")

say_hello()  # نفس النتيجة السابقة تماماً!
