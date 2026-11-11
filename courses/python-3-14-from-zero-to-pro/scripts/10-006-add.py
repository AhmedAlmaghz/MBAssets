#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-006-add.py
"""

# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)
