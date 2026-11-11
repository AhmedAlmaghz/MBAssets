#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-026-__init__.py
"""

from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_salary(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.calculate_salary():.2f} ر.س"

class FullTimeEmployee(Employee):
    def __init__(self, name: str, monthly_salary: float):
        super().__init__(name)
        self.monthly_salary = monthly_salary

    def calculate_salary(self) -> float:
        return self.monthly_salary

class FreelanceEmployee(Employee):
    def __init__(self, name: str, hourly_rate: float, hours_worked: float):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_salary(self) -> float:
        return self.hourly_rate * self.hours_worked

class Payroll:
    def __init__(self):
        self.employees: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def total_payroll(self) -> float:
        return sum(emp.calculate_salary() for emp in self.employees)

    def print_report(self) -> None:
        print("📋 تقرير الرواتب:")
        for emp in self.employees:
            print(f"  {emp}")
        print(f"الإجمالي: {self.total_payroll():.2f} ر.س")

payroll = Payroll()
payroll.add_employee(FullTimeEmployee("أحمد", 8000))
payroll.add_employee(FreelanceEmployee("سارة", 50, 120))
payroll.print_report()

# 📋 تقرير الرواتب:
#   أحمد: 8000.00 ر.س
#   سارة: 6000.00 ر.س
# الإجمالي: 14000.00 ر.س
