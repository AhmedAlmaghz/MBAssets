#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-022-is_palindrome.py
"""

# دوال للاختبار (ضعها في ملف منفصل src/utils.py)
def is_palindrome(text: str) -> bool:
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def calculate_average(numbers: list) -> float:
    if not numbers:
        raise ValueError("القائمة لا يمكن أن تكون فارغة")
    return sum(numbers) / len(numbers)

def is_even(n: int) -> bool:
    return n % 2 == 0

# الاختبارات
import pytest
from utils import is_palindrome, calculate_average, is_even

def test_is_palindrome_true_case():
    assert is_palindrome("مدم") is True

def test_is_palindrome_false_case():
    assert is_palindrome("بيت") is False

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("أ") is True

def test_calculate_average_empty_list_raises_error():
    with pytest.raises(ValueError):
        calculate_average([])

@pytest.mark.parametrize("n, expected", [
    (2, True), (3, False), (0, True), (-4, True), (-7, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected
