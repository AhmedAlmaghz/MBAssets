#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-006-ex16_2.py
"""

# الطريقة 1: قراءة الملف بالكامل كنص واحد
with open("diary.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)

# الطريقة 2: قراءة سطر واحد فقط
with open("diary.txt", "r", encoding="utf-8") as file:
    first_line: str = file.readline()
    print(first_line)  # "اليوم الأول: بدأت تعلم Python.\n"

# الطريقة 3: قراءة كل الأسطر كقائمة
with open("diary.txt", "r", encoding="utf-8") as file:
    lines: list[str] = file.readlines()
    print(lines)
    # ['اليوم الأول: ...\n', 'اليوم الثاني: ...\n', ...]

# الطريقة 4 (الأفضل للملفات الكبيرة): التكرار المباشر سطرًا بسطر
with open("diary.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # .strip() لإزالة \n الزائدة
