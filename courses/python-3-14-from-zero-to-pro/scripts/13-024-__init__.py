#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-024-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner           # Public - وصول حر من أي مكان
        self._balance = balance      # Protected (اصطلاحاً) - "استخدم بحذر"
        self.__pin = "1234"          # Private (اصطلاحاً) - "لا تلمس من خارج الفئة"
