#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-024-data.py
"""

data = json.loads(json_string)
restored_product = Product(**data)  # فك تغليف القاموس كمعاملات
print(restored_product)  # Product(name='لابتوب', price=1200.0)
