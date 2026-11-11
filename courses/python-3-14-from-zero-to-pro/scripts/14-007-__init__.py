#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-007-__init__.py
"""

class Vehicle:
    def __init__(self, brand: str):
        self.brand = brand

    def describe(self) -> str:
        return f"مركبة من {self.brand}"

class Car(Vehicle):
    def __init__(self, brand: str, doors: int):
        super().__init__(brand)
        self.doors = doors

    def describe(self) -> str:
        return f"{super().describe()}, عدد الأبواب: {self.doors}"

class SportsCar(Car):
    def __init__(self, brand: str, doors: int, top_speed: int):
        super().__init__(brand, doors)
        self.top_speed = top_speed

    def describe(self) -> str:
        return f"{super().describe()}, السرعة القصوى: {self.top_speed} كم/س"

ferrari = SportsCar("Ferrari", 2, 340)
print(ferrari.describe())
# مركبة من Ferrari, عدد الأبواب: 2, السرعة القصوى: 340 كم/س

print(isinstance(ferrari, Vehicle))   # True
print(isinstance(ferrari, Car))       # True
print(isinstance(ferrari, SportsCar)) # True
