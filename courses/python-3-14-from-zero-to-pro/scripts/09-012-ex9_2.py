#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-012-ex9_2.py
"""

# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح
