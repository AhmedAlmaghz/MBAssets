#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-031-__init__.py
"""

class Person:
    species = "Homo sapiens"

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self) -> None:
        print(f"مرحباً، اسمي {self.name} وعمري {self.age} سنة.")

p1 = Person("خالد", 30)
p2 = Person("منى", 25)

p1.introduce()  # مرحباً، اسمي خالد وعمري 30 سنة.
p2.introduce()  # مرحباً، اسمي منى وعمري 25 سنة.

print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens
