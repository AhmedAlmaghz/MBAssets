#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.2: المفاهيم الأساسية للاختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-001-test_addition.py
"""

def test_addition():
    # Arrange (التحضير)
    a = 5
    b = 3

    # Act (التنفيذ)
    result = a + b

    # Assert (التحقق)
    assert result == 8
