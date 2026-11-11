#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-028-demo_generator.py
"""

def demo_generator():
    print("البداية")
    yield 1
    print("بعد العنصر الأول")
    yield 2
    print("بعد العنصر الثاني")
    yield 3
    print("النهاية")

gen = demo_generator()  # لا شيء يُطبع بعد! الدالة لم تبدأ التنفيذ بعد

print("--- أول next ---")
print(next(gen))  # البداية ثم 1

print("--- ثاني next ---")
print(next(gen))  # بعد العنصر الأول ثم 2

print("--- ثالث next ---")
print(next(gen))  # بعد العنصر الثاني ثم 3
