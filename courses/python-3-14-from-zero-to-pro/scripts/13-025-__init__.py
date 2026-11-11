#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-025-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # "تلميح" أنه داخلي، لكن لا شيء يمنع الوصول فعلياً

account = BankAccount(1000)
print(account._balance)  # 1000 -- يعمل! Python لا تمنعك، لكنها تُعتبر ممارسة سيئة
