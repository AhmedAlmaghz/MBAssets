#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-018-__init__.py
"""

class Pizza:
    def __init__(self, toppings: list[str]):
        self.toppings = toppings

    def instance_method(self):
        return f"هذه بيتزا بإضافات: {self.toppings}"

    @classmethod
    def margherita(cls) -> "Pizza":
        return cls(["جبنة", "طماطم"])  # Alternative constructor

    @staticmethod
    def calculate_cooking_time(size_cm: int) -> int:
        return size_cm // 2  # منطق عام لا يحتاج بيانات بيتزا معينة

p1 = Pizza.margherita()
print(p1.toppings)  # ['جبنة', 'طماطم']
print(Pizza.calculate_cooking_time(30))  # 15
