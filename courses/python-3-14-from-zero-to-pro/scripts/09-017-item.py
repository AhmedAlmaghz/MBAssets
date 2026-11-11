#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-017-item.py
"""

colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)
