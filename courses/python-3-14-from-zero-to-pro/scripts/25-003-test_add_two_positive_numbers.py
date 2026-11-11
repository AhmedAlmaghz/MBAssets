#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-003-test_add_two_positive_numbers.py
"""

# tests/test_calculator.py
from calculator import add, divide

def test_add_two_positive_numbers():
    result = add(2, 3)
    assert result == 5

def test_add_negative_numbers():
    result = add(-2, -3)
    assert result == -5

def test_divide_normal_case():
    result = divide(10, 2)
    assert result == 5
