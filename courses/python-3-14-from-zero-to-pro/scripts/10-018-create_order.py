#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-018-create_order.py
"""

# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))  
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))  
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط
