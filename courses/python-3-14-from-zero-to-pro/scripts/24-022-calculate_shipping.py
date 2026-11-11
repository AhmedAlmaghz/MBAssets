#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-022-calculate_shipping.py
"""

# 2
# d -> user_data
# tmp -> temporary_file_path
# lst2 -> filtered_products
# flag1 -> is_user_verified

# 3
HEAVY_PACKAGE_THRESHOLD_KG = 20
HEAVY_PACKAGE_RATE = 5.5
STANDARD_RATE = 3.2

def calculate_shipping(weight: float) -> float:
    if weight > HEAVY_PACKAGE_THRESHOLD_KG:
        return weight * HEAVY_PACKAGE_RATE
    return weight * STANDARD_RATE
