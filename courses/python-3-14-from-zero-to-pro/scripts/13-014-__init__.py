#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-014-__init__.py
"""

class Employee:
    company_name = "شركة التقنية المتقدمة"  # مشتركة للجميع
    total_employees = 0  # عداد مشترك

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1  # تعديل صحيح عبر اسم الفئة

emp1 = Employee("أحمد", 8000)
emp2 = Employee("سارة", 9500)
emp3 = Employee("عمر", 7500)

print(f"عدد الموظفين: {Employee.total_employees}")  # 3
print(f"الشركة: {emp1.company_name}")  # شركة التقنية المتقدمة
