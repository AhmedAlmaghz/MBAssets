#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-029-get_first_element.py
"""

from typing import TypeVar

T = TypeVar("T")

def get_first_element(items: list[T]) -> T:
    return items[0]

numbers: list[int] = [10, 20, 30]
first_number = get_first_element(numbers)  # mypy يستنتج: int

words: list[str] = ["كتاب", "قلم"]
first_word = get_first_element(words)  # mypy يستنتج: str
