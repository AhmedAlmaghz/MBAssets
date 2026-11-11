#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-011-__init__.py
"""

# ❌ ينتهك SRP: صنف واحد يفعل أشياء كثيرة غير مرتبطة
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

    def save_to_file(self, path: str) -> None:  # مسؤولية مختلفة: التخزين!
        with open(path, "w") as f:
            f.write(str(self.data))

    def send_email(self, address: str) -> None:  # مسؤولية مختلفة: الاتصال!
        print(f"إرسال التقرير إلى {address}")

# ✅ يتبع SRP: كل صنف مسؤول عن شيء واحد
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

class ReportStorage:
    @staticmethod
    def save_to_file(report: Report, path: str) -> None:
        with open(path, "w") as f:
            f.write(str(report.data))

class ReportMailer:
    @staticmethod
    def send_email(report: Report, address: str) -> None:
        print(f"إرسال التقرير إلى {address}")
