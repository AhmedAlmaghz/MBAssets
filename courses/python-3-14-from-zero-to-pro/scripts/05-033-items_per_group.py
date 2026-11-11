#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-033-items_per_group.py
"""

total_items: int = 10
groups: int = 3
items_per_group = total_items / groups  # 3.333... وليس 3

# إذا أردت عددًا صحيحًا من المجموعات:
items_per_group = total_items // groups  # 3  ← استخدم القسمة الصحيحة
