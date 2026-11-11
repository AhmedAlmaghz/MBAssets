#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-035-add.py
"""

import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()
