#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-028-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Getter - تُستدعى عند قراءة account.balance"""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Setter - تُستدعى عند account.balance = value"""
        if value < 0:
            raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
        self._balance = value

account = BankAccount(1000)
print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

account.balance = 1500   # يستدعي الـ setter تلقائياً
print(account.balance)   # 1500

# account.balance = -500  
# 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!
