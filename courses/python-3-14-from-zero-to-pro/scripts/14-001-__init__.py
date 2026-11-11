#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.1: لماذا الوراثة أحد أهم اختراعات البرمجة الكائنية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-001-__init__.py
"""

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def display_info(self) -> str:
        return f"{self.name}: {self.price} ر.س"

class Book(Product):  # Book ترث من Product
    def __init__(self, name: str, price: float, author: str):
        super().__init__(name, price)  # استدعاء __init__ من الفئة الأساسية
        self.author = author

    def display_info(self) -> str:
        return f"{super().display_info()} - تأليف: {self.author}"

book = Book("بايثون للمحترفين", 120, "محمد أحمد")
print(book.display_info())
# بايثون للمحترفين: 120 ر.س - تأليف: محمد أحمد
