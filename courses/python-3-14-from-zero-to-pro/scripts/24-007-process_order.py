#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-007-process_order.py
"""

# ❌ سيء: دالة واحدة تفعل أشياء كثيرة جدًا
def process_order(order_data):
    # التحقق من الصحة
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر")

    # حساب الإجمالي
    total = sum(item["price"] * item["qty"] for item in order_data["items"])

    # تطبيق الخصم
    if order_data.get("coupon"):
        total *= 0.9

    # حفظ في قاعدة البيانات
    print(f"حفظ الطلب بإجمالي {total}")

    # إرسال بريد إلكتروني
    print("إرسال تأكيد عبر البريد...")

    return total

# ✅ جيد: كل دالة مسؤولة عن شيء واحد فقط
def validate_order(order_data: dict) -> None:
    """التحقق من صحة بيانات الطلب."""
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر في الطلب")

def calculate_total(items: list[dict]) -> float:
    """حساب إجمالي سعر العناصر."""
    return sum(item["price"] * item["qty"] for item in items)

def apply_discount(total: float, has_coupon: bool) -> float:
    """تطبيق خصم 10% إذا وُجد كوبون."""
    return total * 0.9 if has_coupon else total

def save_order(total: float) -> None:
    """حفظ الطلب في قاعدة البيانات."""
    print(f"حفظ الطلب بإجمالي {total}")

def send_confirmation_email() -> None:
    """إرسال بريد تأكيد الطلب."""
    print("إرسال تأكيد عبر البريد...")

def process_order(order_data: dict) -> float:
    """تنسيق عملية معالجة الطلب الكاملة."""
    validate_order(order_data)
    total = calculate_total(order_data["items"])
    total = apply_discount(total, bool(order_data.get("coupon")))
    save_order(total)
    send_confirmation_email()
    return total
