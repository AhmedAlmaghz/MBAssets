#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-026-compare_inventories.py
"""

def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}
