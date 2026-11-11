#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-024-frozen.py
"""

# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}
