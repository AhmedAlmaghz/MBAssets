#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-014-add_item.py
"""

# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!
