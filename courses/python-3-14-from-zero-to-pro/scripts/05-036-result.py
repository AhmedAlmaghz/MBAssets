#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.16: مشروع تطبيقي: «الحاسبة العلمية التفاعلية»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-036-result.py
"""

# scientific_calculator.py
"""حاسبة علمية تفاعلية بسيطة من سطر الأوامر."""

print("=" * 40)
print("🧮  الحاسبة العلمية التفاعلية")
print("=" * 40)

# استقبال المدخلات
num1: float = float(input("أدخل الرقم الأول: "))
operator: str = input("أدخل العملية (+ - * / // % **): ").strip()
num2: float = float(input("أدخل الرقم الثاني: "))

# معالجة حالة خاصة: القسمة على صفر
if operator in ("/", "//", "%") and num2 == 0:
    print("\n❌ خطأ: لا يمكن القسمة على صفر!")
else:
    result: float = 0.0
    valid_operation: bool = True

    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        result = num1 / num2
    elif operator == "//":
        result = num1 // num2
    elif operator == "%":
        result = num1 % num2
    elif operator == "**":
        result = num1 ** num2
    else:
        valid_operation = False
        print(f"\n❌ خطأ: العملية '{operator}' غير معروفة!")

    if valid_operation:
        print("\n" + "=" * 40)
        print(f"📊 النتيجة: {num1} {operator} {num2} = {round(result, 4)}")
        print("=" * 40)
