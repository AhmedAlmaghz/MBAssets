#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-009-defaults.py
"""

# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}
