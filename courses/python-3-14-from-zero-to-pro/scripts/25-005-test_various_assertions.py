#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-005-test_various_assertions.py
"""

def test_various_assertions():
    assert 5 > 3                           # مقارنة
    assert "hello" in "hello world"        # الاحتواء
    assert [1, 2, 3] == [1, 2, 3]           # تساوي القوائم
    assert isinstance(5, int)               # النوع
    assert not False                        # النفي
    assert {"a": 1} == {"a": 1}             # تساوي القواميس
