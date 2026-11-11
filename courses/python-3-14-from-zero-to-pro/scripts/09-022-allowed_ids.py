#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-022-allowed_ids.py
"""

# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)
