#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-004-person2.py
"""

# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami
