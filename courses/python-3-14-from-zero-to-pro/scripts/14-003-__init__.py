#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-003-__init__.py
"""

class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def display(self) -> str:
        return f"{self.name}: {self.salary} ر.س"

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)  # استدعاء __init__ الأساسي أولاً
        self.team_size = team_size

    def display(self) -> str:
        base_info = super().display()  # استدعاء display الأساسي، ثم إضافة عليه
        return f"{base_info} (يدير فريق من {self.team_size} أشخاص)"

manager = Manager("نورة", 15000, 8)
print(manager.display())
# نورة: 15000 ر.س (يدير فريق من 8 أشخاص)
