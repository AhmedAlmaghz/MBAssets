#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-011-make_counter.py
"""

def make_counter():
    """تُرجع دالة تعد تصاعدياً في كل استدعاء."""
    count = 0

    def increment():
        nonlocal count  # بدون هذه الكلمة، سيحدث UnboundLocalError
        count += 1
        return count

    return increment

counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # عداد جديد مستقل تماماً
print(counter2())  # 1  -- لا علاقة له بـ counter1!
