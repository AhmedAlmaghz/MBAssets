#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-018-product.py
"""

"""نماذج بيانات نظام إدارة المخزون."""

from dataclasses import dataclass

@dataclass
class Product:
    """يمثّل منتجًا في المخزون."""

    sku: str
    name: str
    price: float
    quantity: int

class ProductNotFoundError(Exception):
    """يُرفع عند البحث عن منتج غير موجود."""
    pass

class InsufficientStockError(Exception):
    """يُرفع عند محاولة بيع كمية أكبر من المتوفر."""
    pass

class DuplicateSKUError(Exception):
    """يُرفع عند محاولة إضافة منتج برمز SKU موجود مسبقًا."""
    pass
