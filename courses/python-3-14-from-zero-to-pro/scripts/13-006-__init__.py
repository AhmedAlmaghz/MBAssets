#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-006-__init__.py
"""

class Car:
    """تمثل سيارة بخصائصها الأساسية."""

    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage: int = 0  # قيمة ابتدائية افتراضية

    def drive(self, distance: int) -> None:
        """تُحرّك السيارة مسافة معينة، وتُحدّث العداد."""
        self.mileage += distance
        print(f"🚗 قطعت {self.brand} {distance} كم. المسافة الكلية: {self.mileage} كم")

    def display_info(self) -> str:
        return f"{self.year} {self.brand} {self.model} - {self.mileage} كم"

my_car = Car("تويوتا", "كورولا", 2023)
my_car.drive(150)
my_car.drive(80)
print(my_car.display_info())
