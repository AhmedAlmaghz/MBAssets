#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-024-get_shipping_cost.py
"""

def get_shipping_cost(order):
    if order is not None:
        if order.weight > 0:
            if order.destination == "local":
                return order.weight * 2
            else:
                return order.weight * 5
    return 0
