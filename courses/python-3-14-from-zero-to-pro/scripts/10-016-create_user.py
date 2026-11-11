#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-016-create_user.py
"""

def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)
