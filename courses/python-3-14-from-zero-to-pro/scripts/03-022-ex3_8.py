#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث: المتغيرات - أول لبنة
القسم 3.8: تحويل الأنواع (Type Casting)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 03-022-ex3_8.py
"""

# تحويل من str إلى int
age_text: str = "25"
age_number: int = int(age_text)
print(age_number + 5)  # 30

# تحويل من int إلى str
score: int = 95
score_text: str = str(score)
print("درجتك: " + score_text)  # درجتك: 95

# تحويل من str إلى float
price_text: str = "19.99"
price: float = float(price_text)

# تحويل إلى bool
print(bool(0))      # False
print(bool(1))      # True
print(bool(""))     # False (نص فارغ = False)
print(bool("لا"))   # True  (أي نص غير فارغ = True، حتى لو كان "لا"!)
