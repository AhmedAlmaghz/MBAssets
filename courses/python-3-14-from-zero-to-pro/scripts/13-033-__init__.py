#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-033-__init__.py
"""

class ShoppingCart:
    def __init__(self):
        self._items: list[dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self._items.append({"name": name, "price": price, "quantity": quantity})
        print(f"✅ أُضيف {quantity}x {name} إلى السلة.")

    def total_price(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self._items)

    @property
    def item_count(self) -> int:
        """إجمالي عدد القطع (مجموع الكميات)."""
        return sum(item["quantity"] for item in self._items)

    def __len__(self) -> int:
        """عدد المنتجات المختلفة في السلة."""
        return len(self._items)

    def __str__(self) -> str:
        lines = [f"🛒 سلة التسوق ({len(self)} منتج مختلف):"]
        for item in self._items:
            subtotal = item["price"] * item["quantity"]
            lines.append(f"  - {item['name']}: {item['quantity']} × {item['price']} = {subtotal}")
        lines.append(f"الإجمالي: {self.total_price()} ر.س")
        return "\n".join(lines)

cart = ShoppingCart()
cart.add_item("لابتوب", 3500, 1)
cart.add_item("ماوس", 50, 2)
cart.add_item("كيبورد", 150, 1)

print(cart)
print(f"\nعدد القطع الكلي: {cart.item_count}")  # 4
print(f"عدد المنتجات المختلفة: {len(cart)}")    # 3
