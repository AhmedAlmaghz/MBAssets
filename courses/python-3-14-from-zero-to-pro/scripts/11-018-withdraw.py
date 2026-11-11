#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-018-withdraw.py
"""

class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")
