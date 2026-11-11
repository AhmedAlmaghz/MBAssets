#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-020-empty_manager.py
"""

"""Fixtures مشتركة متاحة تلقائيًا لكل ملفات الاختبار في هذا المجلد."""

import pytest

from inventory.inventory_manager import InventoryManager
from inventory.models import Product

@pytest.fixture
def empty_manager():
    """مدير مخزون فارغ تمامًا."""
    return InventoryManager()

@pytest.fixture
def manager_with_products(empty_manager):
    """مدير مخزون مُعبّأ بمنتجات تجريبية جاهزة للاختبارات."""
    empty_manager.add_product(Product(sku="LAP001", name="لابتوب", price=3500, quantity=10))
    empty_manager.add_product(Product(sku="MOU001", name="ماوس", price=50, quantity=100))
    empty_manager.add_product(Product(sku="KEY001", name="لوحة مفاتيح", price=120, quantity=3))
    return empty_manager
