#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-016-__init__.py
"""

class Employee:
    company_name = "شركة التقنية"
    total_employees = 0

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1

    @classmethod
    def get_total_employees(cls) -> int:
        """تستقبل cls (الفئة نفسها) بدلاً من self (كائن محدد)."""
        return cls.total_employees

    @classmethod
    def from_string(cls, data_string: str) -> "Employee":
        """Alternative Constructor - نمط شائع جداً لإنشاء كائنات من صيغ بيانات مختلفة."""
        name, salary = data_string.split(",")
        return cls(name.strip(), float(salary.strip()))

emp1 = Employee("أحمد", 8000)
emp2 = Employee.from_string("سارة, 9500")  # إنشاء من نص!

print(emp2.name, emp2.salary)              # سارة 9500.0
print(Employee.get_total_employees())      # 2
