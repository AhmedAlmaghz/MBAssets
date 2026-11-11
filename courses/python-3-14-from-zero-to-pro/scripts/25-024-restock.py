#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-024-restock.py
"""

# إضافة إلى inventory_manager.py
def restock(self, sku: str, quantity: int) -> None:
    """زيادة كمية منتج موجود في المخزون.

    Raises:
        ProductNotFoundError: إذا لم يوجد المنتج.
        ValueError: إذا كانت الكمية المُضافة غير موجبة.
    """
    if quantity <= 0:
        raise ValueError("كمية التزويد يجب أن تكون موجبة")

    product = self.get_product(sku)  # يرفع ProductNotFoundError تلقائيًا إن لم يوجد
    product.quantity += quantity

# الاختبارات
class TestRestock:
    def test_restock_increases_quantity(self, manager_with_products):
        manager_with_products.restock("KEY001", 10)
        product = manager_with_products.get_product("KEY001")
        assert product.quantity == 13

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -50])
    def test_restock_non_positive_raises_error(self, manager_with_products, invalid_quantity):
        with pytest.raises(ValueError):
            manager_with_products.restock("KEY001", invalid_quantity)

    def test_restock_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.restock("NOTEXIST", 10)
