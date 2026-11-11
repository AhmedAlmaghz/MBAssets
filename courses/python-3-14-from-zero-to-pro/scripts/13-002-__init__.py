#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.1: لماذا تُعتبر البرمجة الكائنية نقلة نوعية في التفكير؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-002-__init__.py
"""

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.grades: list[float] = []

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student1 = Student("أحمد", 20)
student1.add_grade(85)
student1.add_grade(90)
print(student1.average_grade())  # 87.5
