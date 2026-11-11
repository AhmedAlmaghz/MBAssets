#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.8: `else` في الحلقات: ميزة نادرًا ما تُشرح بوضوح

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-013-ex7_8.py
"""

for number in range(1, 6):
    if number == 3:
        break
    print(number)
else:
    print("اكتملت الحلقة بنجاح!")  # لن تُطبع!
