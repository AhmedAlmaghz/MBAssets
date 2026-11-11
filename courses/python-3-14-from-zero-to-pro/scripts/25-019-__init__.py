#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-019-__init__.py
"""

"""منطق إدارة المخزون: إضافة، بيع، واستعلام عن المنتجات."""

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class InventoryManager:
    """يدير مجموعة من المنتجات داخل المخزون."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """إضافة منتج جديد إلى المخزون.

        Raises:
            DuplicateSKUError: إذا كان رمز SKU مستخدمًا مسبقًا.
        """
        if product.sku in self._products:
            raise DuplicateSKUError(f"المنتج {product.sku} موجود بالفعل")
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        """الحصول على منتج بواسطة رمزه.

        Raises:
            ProductNotFoundError: إذا لم يوجد منتج بهذا الرمز.
        """
        if sku not in self._products:
            raise ProductNotFoundError(f"المنتج {sku} غير موجود")
        return self._products[sku]

    def sell(self, sku: str, quantity: int) -> float:
        """بيع كمية من منتج، وإرجاع الإيراد الناتج.

        Raises:
            ProductNotFoundError: إذا لم يوجد المنتج.
            ValueError: إذا كانت الكمية المطلوبة غير موجبة.
            InsufficientStockError: إذا تجاوزت الكمية المطلوبة المخزون المتاح.
        """
        if quantity <= 0:
            raise ValueError("الكمية يجب أن تكون موجبة")

        product = self.get_product(sku)

        if quantity > product.quantity:
            raise InsufficientStockError(
                f"الكمية المتوفرة ({product.quantity}) أقل من المطلوبة ({quantity})"
            )

        product.quantity -= quantity
        return product.price * quantity

    def total_inventory_value(self) -> float:
        """حساب القيمة الإجمالية لكل المخزون الحالي."""
        return sum(p.price * p.quantity for p in self._products.values())

    def low_stock_products(self, threshold: int = 5) -> list[Product]:
        """إرجاع قائمة المنتجات التي كميتها أقل من أو تساوي حدًا معيّنًا."""
        return [p for p in self._products.values() if p.quantity <= threshold]
