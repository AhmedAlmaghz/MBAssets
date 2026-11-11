#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-047-find_product.py
"""

def find_product(product_id: int, catalog: dict[int, str]) -> str:
    return catalog.get(product_id)

def get_discount(price: float, percent: int) -> float:
    return price - (price * percent)

products = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)
print(name.upper())
