#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.2: المفاهيم الأساسية: Class و Object

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-003-__init__.py
"""

class Dog:
    """قالب (Class) يصف أي كلب."""
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} يقول: واق واق!"

# إنشاء كائنات (Objects / Instances) من نفس القالب
dog1 = Dog("ريكس", "جيرمن شيبرد")
dog2 = Dog("لونا", "بودل")

print(dog1.bark())  # ريكس يقول: واق واق!
print(dog2.bark())  # لونا يقول: واق واق!
print(dog1.name)    # ريكس
print(dog2.breed)   # بودل
