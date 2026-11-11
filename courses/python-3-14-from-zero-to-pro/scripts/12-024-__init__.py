#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-024-__init__.py
"""

class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property  # decorator مدمج - يجعل method يُستخدم كخاصية
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975 -- بدون أقواس استدعاء!

from functools import lru_cache

@lru_cache(maxsize=None)  # تذكر - caching تلقائي! (كتبناه يدوياً في الفصل 10)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # سريع جداً بفضل lru_cache، رغم أن fibonacci بطيئة أصلاً
