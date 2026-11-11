#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-013-add_item.py
"""

# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!
