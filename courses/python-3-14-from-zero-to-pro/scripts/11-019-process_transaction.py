#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-019-process_transaction.py
"""

class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")
