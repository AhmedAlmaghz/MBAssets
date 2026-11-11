#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.8: نموذج عملي: قبل وبعد (Refactoring كامل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-020-calculate_type_a_discount.py
"""

# ===== بعد: نظيف، واضح، وقابل للصيانة =====

TYPE_A = "A"
TYPE_B = "B"

TYPE_A_BULK_THRESHOLD = 100
TYPE_A_BULK_DISCOUNT = 0.9
TYPE_B_STANDARD_DISCOUNT = 0.95

def calculate_type_a_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع A (خصم إضافي للكميات الكبيرة)."""
    if amount > TYPE_A_BULK_THRESHOLD:
        return amount * TYPE_A_BULK_DISCOUNT
    return amount

def calculate_type_b_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع B (خصم ثابت)."""
    return amount * TYPE_B_STANDARD_DISCOUNT

def process_items(items: list[dict]) -> list[dict]:
    """معالجة قائمة عناصر وحساب السعر النهائي لكل منها حسب نوعه.

    Args:
        items: قائمة قواميس، كل منها يحتوي 'id', 'type', 'amt'.

    Returns:
        قائمة قواميس تحتوي 'id' و'final' (السعر النهائي بعد الخصم).
    """
    processed_items = []

    for item in items:
        if item["type"] == TYPE_A:
            final_price = calculate_type_a_discount(item["amt"])
        elif item["type"] == TYPE_B:
            final_price = calculate_type_b_discount(item["amt"])
        else:
            continue  # تخطَّ الأنواع غير المعروفة بدلاً من كسر البرنامج

        processed_items.append({"id": item["id"], "final": final_price})

    return processed_items
