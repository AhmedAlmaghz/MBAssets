#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-032-__init__.py
"""

class Repository[T]:
    """مخزن بسيط يدير مجموعة من العناصر من نوع T."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        return self._items.copy()

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

from typing import Callable
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product_repo = Repository[Product]()
product_repo.add(Product("لابتوب", 1200.0))
product_repo.add(Product("هاتف", 800.0))

cheap_product = product_repo.find(lambda p: p.price < 1000)
print(cheap_product)  # Product(name='هاتف', price=800.0)
