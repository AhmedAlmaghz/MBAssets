#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-019-user_a_purchases.py
"""

user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}
