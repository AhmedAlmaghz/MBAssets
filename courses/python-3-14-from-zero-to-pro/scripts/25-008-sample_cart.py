#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-008-sample_cart.py
"""

import pytest

@pytest.fixture
def sample_cart():
    """Fixture تُجهّز سلة تسوّق تجريبية لاستخدامها في عدة اختبارات."""
    return {
        "items": [
            {"name": "لابتوب", "price": 3500, "qty": 1},
            {"name": "ماوس", "price": 50, "qty": 2},
        ],
    }

def test_cart_has_two_item_types(sample_cart):
    assert len(sample_cart["items"]) == 2

def test_cart_total_quantity(sample_cart):
    total_qty = sum(item["qty"] for item in sample_cart["items"])
    assert total_qty == 3
