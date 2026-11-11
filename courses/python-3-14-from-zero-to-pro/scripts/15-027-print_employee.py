#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.9: القواميس المُهيكَلة: TypedDict

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-027-print_employee.py
"""

from typing import TypedDict

class Employee(TypedDict):
    name: str
    age: int
    salary: float

def print_employee(emp: Employee) -> None:
    print(f"{emp['name']} - {emp['age']} سنة - راتب {emp['salary']} دولار")

employee1: Employee = {
    "name": "ياسر",
    "age": 35,
    "salary": 5000.0,
}

print_employee(employee1)

# استخدام خاطئ يكتشفه mypy:
bad_employee: Employee = {
    "name": "سلمى",
    "age": "ثلاثون",  # ❌ خطأ! يجب أن يكون int وليس str
    "salary": 4500.0,
}
