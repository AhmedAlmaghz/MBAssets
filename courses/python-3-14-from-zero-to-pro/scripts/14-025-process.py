#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-025-process.py
"""

# تمرين 3
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        pass

class CreditCard(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

class PayPal(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

methods: list[PaymentMethod] = [CreditCard(), PayPal()]
for method in methods:
    method.process(100)

# تمرين 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity

p = Product("لابتوب", 3500, 2)
print(p.total())  # 7000
# p.price = 4000  # 💥 FrozenInstanceError
