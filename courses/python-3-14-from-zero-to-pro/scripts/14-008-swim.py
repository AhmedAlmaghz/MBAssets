#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-008-swim.py
"""

class Swimmer:
    def swim(self) -> str:
        return "يسبح 🏊"

class Flyer:
    def fly(self) -> str:
        return "يطير 🦅"

class Duck(Swimmer, Flyer):  # وراثة من فئتين معاً
    def __init__(self, name: str):
        self.name = name

duck = Duck("دونالد")
print(duck.swim())  # يسبح 🏊
print(duck.fly())   # يطير 🦅
