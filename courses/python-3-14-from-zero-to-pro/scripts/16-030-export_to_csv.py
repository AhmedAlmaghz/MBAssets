#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-030-export_to_csv.py
"""

from dataclasses import dataclass, asdict, fields
import csv
from pathlib import Path

@dataclass
class Employee:
    name: str
    department: str
    salary: float

def export_to_csv(employees: list[Employee], file_path: Path) -> None:
    if not employees:
        return
    fieldnames = [f.name for f in fields(Employee)]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for emp in employees:
            writer.writerow(asdict(emp))

employees = [
    Employee("ياسمين", "تسويق", 6000.0),
    Employee("عمر", "هندسة", 8500.0),
]

export_to_csv(employees, Path("employees.csv"))
