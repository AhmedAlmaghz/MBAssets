#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-005-safe_divide.py
"""

def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0
