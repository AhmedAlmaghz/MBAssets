#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-002-__init__.py
"""

class Animal:  # Parent / Base / Super Class
    def __init__(self, name: str):
        self.name = name

    def make_sound(self) -> str:
        return "صوت ما..."

    def __str__(self) -> str:
        return f"{self.name} يصدر صوت: {self.make_sound()}"

class Dog(Animal):  # Child / Derived / Sub Class
    def make_sound(self) -> str:  # Override - استبدال السلوك
        return "واق واق!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "مياو!"

dog = Dog("ريكس")
cat = Cat("لونا")

print(dog)  # ريكس يصدر صوت: واق واق!
print(cat)  # لونا يصدر صوت: مياو!
