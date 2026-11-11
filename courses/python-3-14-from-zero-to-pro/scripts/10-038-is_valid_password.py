#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-038-is_valid_password.py
"""

# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]
