#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-030-orders.py
"""

from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")
