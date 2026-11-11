#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-004-test_add_fails_example.py
"""

def test_add_fails_example():
    result = add(2, 2)
    assert result == 5  # سيفشل عمدًا لنرى رسالة الخطأ
