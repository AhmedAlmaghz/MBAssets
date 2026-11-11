#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-012-__init__.py
"""

class Dog:
    species = "Canis familiaris"  # Class Attribute - مشتركة بين كل الكلاب

    def __init__(self, name: str):
        self.name = name  # Instance Attribute - خاصة بكل كلب

    def __str__(self) -> str:
        return f"{self.name} ({self.species})"

dog1 = Dog("ريكس")
dog2 = Dog("لونا")

print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris -- نفس القيمة للجميع

# الوصول من الفئة مباشرة أيضاً ممكن
print(Dog.species)   # Canis familiaris
