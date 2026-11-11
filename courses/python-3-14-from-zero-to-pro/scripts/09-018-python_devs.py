#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-018-python_devs.py
"""

python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}
