#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.9: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-040-make_adder.py
"""

def make_adder(x):
    def add(y):
        return x + y
    return add

add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))
