#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-023-sample_products.py
"""

# 4
import pytest

@pytest.fixture
def sample_products():
    return [
        {"name": "أ", "price": 100},
        {"name": "ب", "price": 200},
        {"name": "ج", "price": 300},
    ]

def average_price(products: list[dict]) -> float:
    return sum(p["price"] for p in products) / len(products)

def test_average_price(sample_products):
    assert average_price(sample_products) == 200

# 5
from datetime import datetime
from unittest.mock import patch

def get_current_year() -> int:
    return datetime.now().year

def test_get_current_year_with_fixed_date():
    fixed_date = datetime(2030, 1, 1)
    with patch("__main__.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_date
        assert get_current_year() == 2030
