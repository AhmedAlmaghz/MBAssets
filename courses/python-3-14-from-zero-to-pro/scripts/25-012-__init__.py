#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-012-__init__.py
"""

# src/bank_account.py
class InsufficientFundsError(Exception):
    """يُرفع عند محاولة سحب مبلغ أكبر من الرصيد المتاح."""
    pass

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("مبلغ الإيداع يجب أن يكون موجبًا")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFundsError("الرصيد غير كافٍ")
        self.balance -= amount
