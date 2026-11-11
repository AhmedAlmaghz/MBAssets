#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-036-log_calculation.py
"""

calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator
