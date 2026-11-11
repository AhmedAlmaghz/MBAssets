#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-023-product.py
"""

from dataclasses import dataclass, asdict
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)

product_dict: dict[str, object] = asdict(product)
json_string = json.dumps(product_dict, ensure_ascii=False, indent=2)
print(json_string)
# {
#   "name": "لابتوب",
#   "price": 1200.0
# }
