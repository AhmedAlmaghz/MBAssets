#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-004-__init__.py
"""

class Manager(Employee):
    def __init__(self, name, salary, team_size):
        # ⚠️ نسيان super().__init__(name, salary)!
        self.team_size = team_size

manager = Manager("نورة", 15000, 8)
print(manager.name)  
# 💥 AttributeError: 'Manager' object has no attribute 'name'
