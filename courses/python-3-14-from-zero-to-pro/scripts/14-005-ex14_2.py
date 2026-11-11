#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-005-ex14_2.py
"""

print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - Dog هي أيضاً Animal!
print(isinstance(dog, Cat))     # False

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False - العلاقة أحادية الاتجاه
