#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-048-find_product.py
"""

# تمرين 4 - الكود المصحح
def find_product(product_id: int, catalog: dict[int, str]) -> str | None:
    return catalog.get(product_id)  # قد تكون None فعلاً

def get_discount(price: float, percent: int) -> float:
    return price - (price * (percent / 100))  # كانت تُطرح قيمة كبيرة جدًا خطأً منطقيًا

products: dict[int, str] = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)

if name is not None:
    print(name.upper())
else:
    print("المنتج غير موجود")

# تمرين 5
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T | None:
        if self.is_empty():
            return None
        return self._items.pop()

    def peek(self) -> T | None:
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())   # 2
print(stack.peek())  # 1

# تمرين 6
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other: object) -> bool:
        ...

def find_max[T: Comparable](items: list[T]) -> T:
    if not items:
        raise ValueError("القائمة فارغة")
    current_max = items[0]
    for item in items[1:]:
        if current_max < item:
            current_max = item
    return current_max

print(find_max([3, 7, 2, 9, 4]))        # 9
print(find_max(["بطة", "قطة", "أسد"]))  # قطة (حسب الترتيب الأبجدي)
