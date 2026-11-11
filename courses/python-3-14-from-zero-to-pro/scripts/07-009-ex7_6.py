#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.6: `break`: الخروج الفوري من الحلقة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-009-ex7_6.py
"""

for number in range(1, 11):
    if number == 5:
        break  # إيقاف الحلقة بالكامل فور الوصول لـ 5
    print(number)
