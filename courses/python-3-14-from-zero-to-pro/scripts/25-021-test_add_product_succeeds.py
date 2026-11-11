#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-021-test_add_product_succeeds.py
"""

"""اختبارات شاملة لنظام إدارة المخزون."""

import pytest

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class TestAddProduct:
    """اختبارات إضافة منتج جديد."""

    def test_add_product_succeeds(self, empty_manager):
        product = Product(sku="TEST01", name="منتج تجريبي", price=100, quantity=5)
        empty_manager.add_product(product)

        retrieved = empty_manager.get_product("TEST01")
        assert retrieved.name == "منتج تجريبي"

    def test_add_duplicate_sku_raises_error(self, manager_with_products):
        duplicate = Product(sku="LAP001", name="لابتوب آخر", price=4000, quantity=5)

        with pytest.raises(DuplicateSKUError):
            manager_with_products.add_product(duplicate)

class TestGetProduct:
    """اختبارات استرجاع منتج."""

    def test_get_existing_product(self, manager_with_products):
        product = manager_with_products.get_product("MOU001")
        assert product.name == "ماوس"
        assert product.price == 50

    def test_get_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.get_product("NOTEXIST")

class TestSellProduct:
    """اختبارات عملية البيع."""

    def test_sell_reduces_quantity(self, manager_with_products):
        manager_with_products.sell("LAP001", 3)
        remaining = manager_with_products.get_product("LAP001")
        assert remaining.quantity == 7

    def test_sell_returns_correct_revenue(self, manager_with_products):
        revenue = manager_with_products.sell("MOU001", 10)
        assert revenue == 500  # 50 * 10

    def test_sell_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.sell("NOTEXIST", 1)

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -100])
    def test_sell_non_positive_quantity_raises_error(
        self, manager_with_products, invalid_quantity
    ):
        with pytest.raises(ValueError):
            manager_with_products.sell("LAP001", invalid_quantity)

    def test_sell_more_than_available_raises_error(self, manager_with_products):
        with pytest.raises(InsufficientStockError):
            manager_with_products.sell("KEY001", 1000)

    def test_sell_exact_remaining_quantity_succeeds(self, manager_with_products):
        """حالة حدّية: بيع الكمية المتبقية بالضبط يجب أن ينجح (وليس يرفض)."""
        manager_with_products.sell("KEY001", 3)  # الكمية المتوفرة بالضبط
        remaining = manager_with_products.get_product("KEY001")
        assert remaining.quantity == 0

class TestInventoryReports:
    """اختبارات تقارير المخزون الإجمالية."""

    def test_total_inventory_value(self, manager_with_products):
        # (3500*10) + (50*100) + (120*3) = 35000 + 5000 + 360 = 40360
        assert manager_with_products.total_inventory_value() == 40360

    def test_empty_inventory_value_is_zero(self, empty_manager):
        assert empty_manager.total_inventory_value() == 0

    def test_low_stock_products_default_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products()
        skus = [p.sku for p in low_stock]
        assert "KEY001" in skus      # كمية 3، أقل من الحد الافتراضي 5
        assert "LAP001" not in skus  # كمية 10، أعلى من الحد

    def test_low_stock_products_custom_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products(threshold=15)
        skus = [p.sku for p in low_stock]
        assert "LAP001" in skus  # كمية 10، أقل من الحد المخصص 15
