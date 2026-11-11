#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-039-batch_generator.py
"""

def batch_generator(items: list, batch_size: int):
    """يقسّم قائمة إلى دفعات، مفيد عند معالجة بيانات ضخمة على أجزاء."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # إرسال أي عناصر متبقية أقل من batch_size
        yield batch

items = [1, 2, 3, 4, 5, 6, 7]
for batch in batch_generator(items, 3):
    print(batch)

# [1, 2, 3]
# [4, 5, 6]
# [7]
