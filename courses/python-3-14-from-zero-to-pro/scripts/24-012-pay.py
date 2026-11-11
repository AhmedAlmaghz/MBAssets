#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-012-pay.py
"""

from abc import ABC, abstractmethod

# ✅ يتبع OCP: إضافة طريقة دفع جديدة لا تتطلب تعديل أي كود قديم
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...

class CreditCardPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر بطاقة الائتمان"

class PayPalPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر PayPal"

# لإضافة طريقة دفع جديدة، أضف صنفًا جديدًا فقط -- بدون لمس الكود القديم!
class CryptoPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر العملات الرقمية"

def checkout(method: PaymentMethod, amount: float) -> None:
    print(method.pay(amount))

checkout(CreditCardPayment(), 100)
checkout(CryptoPayment(), 50)
