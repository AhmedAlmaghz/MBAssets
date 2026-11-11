#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.6: Parametrize: اختبار حالات متعددة بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-011-test_add_multiple_cases.py
"""

import pytest
from calculator import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected
