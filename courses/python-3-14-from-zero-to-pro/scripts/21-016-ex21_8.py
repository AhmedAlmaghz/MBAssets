#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-016-ex21_8.py
"""

import pandas as pd

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")
print(df.describe())
