#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-022-product.py
"""

from dataclasses import dataclass
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)
json.dumps(product)  # ❌ TypeError: Object of type Product is not JSON serializable
