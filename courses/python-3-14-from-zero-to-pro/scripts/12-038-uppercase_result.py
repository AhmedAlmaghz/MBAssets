#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-038-uppercase_result.py
"""

# تمرين 3
from functools import wraps

def uppercase_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_result
def get_greeting(name: str) -> str:
    return f"hello, {name}"

print(get_greeting("ahmed"))  # HELLO, AHMED

# تمرين 4
def make_bank_account(initial_balance: float):
    balance = initial_balance

    def deposit(amount: float) -> None:
        nonlocal balance
        balance += amount
        print(f"✅ تم إيداع {amount}. الرصيد الآن: {balance}")

    def check_balance() -> float:
        return balance

    return deposit, check_balance

deposit, check_balance = make_bank_account(1000)
deposit(500)                      # ✅ تم إيداع 500. الرصيد الآن: 1500
print(check_balance())            # 1500
deposit(200)                      # ✅ تم إيداع 200. الرصيد الآن: 1700
print(check_balance())            # 1700
