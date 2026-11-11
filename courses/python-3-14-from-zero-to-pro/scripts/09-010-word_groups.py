#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-010-word_groups.py
"""

from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))
