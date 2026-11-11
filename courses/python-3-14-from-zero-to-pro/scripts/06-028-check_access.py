#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.11: الشروط المتداخلة (Nested Conditions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-028-check_access.py
"""

# بدلاً من هذا (متداخل)
def check_access(age, has_id):
    if age >= 18:
        if has_id:
            return "مسموح"
        else:
            return "تحتاج هوية"
    else:
        return "عمر غير كافٍ"

# استخدم هذا (أوضح - Guard Clauses)
def check_access(age, has_id):
    if age < 18:
        return "عمر غير كافٍ"
    if not has_id:
        return "تحتاج هوية"
    return "مسموح"
