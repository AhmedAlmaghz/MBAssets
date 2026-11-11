#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-006-test_divide_by_zero_raises_error.py
"""

import pytest
from calculator import divide

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="لا يمكن القسمة على صفر"):
        divide(10, 0)
