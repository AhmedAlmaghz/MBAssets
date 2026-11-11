#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-008-ex8_5.py
"""

fruits: list[str] = ["تفاح", "موز", "برتقال", "موز"]

# حذف بالقيمة (أول ظهور فقط)
fruits.remove("موز")
print(fruits)  # ['تفاح', 'برتقال', 'موز']

# حذف بالفهرس، وإعادة القيمة المحذوفة
removed_item: str = fruits.pop(0)
print(removed_item)  # تفاح
print(fruits)         # ['برتقال', 'موز']

# pop() بدون فهرس يحذف آخر عنصر
last_item: str = fruits.pop()
print(last_item)  # موز

# حذف بالفهرس دون الحاجة للقيمة المحذوفة
del fruits[0]

# إفراغ القائمة بالكامل
fruits.clear()
print(fruits)  # []
