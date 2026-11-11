#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-007-test_float_addition_naive.py
"""

def test_float_addition_naive():
    assert 0.1 + 0.2 == 0.3  # ❌ قد يفشل! بسبب دقة الفاصلة العائمة

def test_float_addition_correct():
    assert 0.1 + 0.2 == pytest.approx(0.3)  # ✅ يتحقق بتسامح معقول من الفروق الدقيقة جدًا
