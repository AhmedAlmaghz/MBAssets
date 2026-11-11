#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-026-__init__.py
"""

class BankAccount:
    def __init__(self, pin: str):
        self.__pin = pin

    def verify_pin(self, attempt: str) -> bool:
        return self.__pin == attempt

account = BankAccount("1234")
print(account.verify_pin("1234"))  # True

# print(account.__pin)  
# 💥 AttributeError: 'BankAccount' object has no attribute '__pin'
