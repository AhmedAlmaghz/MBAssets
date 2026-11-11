#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-008-outer_function.py
"""

def outer_function():
    message = "مرحباً من outer!"

    def inner_function():
        print(message)  # inner تصل لمتغير outer بفضل قاعدة LEGB (الفصل 10)

    inner_function()

outer_function()  # مرحباً من outer!
