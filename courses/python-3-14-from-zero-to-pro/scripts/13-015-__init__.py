#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-015-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:  # instance method
        self.balance += amount
