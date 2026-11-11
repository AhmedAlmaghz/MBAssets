#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-031-__init__.py
"""

class Box[T]:
    """صندوق يحتفظ بعنصر واحد من أي نوع."""

    def __init__(self, content: T) -> None:
        self.content = content

    def get(self) -> T:
        return self.content

    def __repr__(self) -> str:
        return f"Box({self.content!r})"

int_box = Box(42)           # mypy يستنتج: Box[int]
str_box = Box("مرحبًا")      # mypy يستنتج: Box[str]

print(int_box.get())  # 42
print(str_box.get())  # مرحبًا

value: int = int_box.get()  # صحيح، mypy يعرف أن الناتج int
