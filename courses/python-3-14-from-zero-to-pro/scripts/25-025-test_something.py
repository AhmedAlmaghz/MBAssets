#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.15: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-025-test_something.py
"""

# ===== الأساسيات =====
def test_something():                    # يجب أن يبدأ بـ test_
    assert actual == expected

# ===== اختبار الاستثناءات =====
import pytest
with pytest.raises(ValueError):
    risky_function()
with pytest.raises(ValueError, match="رسالة محددة"):
    risky_function()

# ===== الأعداد العشرية =====
assert 0.1 + 0.2 == pytest.approx(0.3)

# ===== Fixtures =====
@pytest.fixture
def my_data():
    return {"key": "value"}

def test_uses_fixture(my_data):
    assert my_data["key"] == "value"

@pytest.fixture
def resource_with_cleanup():
    resource = setup()
    yield resource
    teardown(resource)

# ===== Parametrize =====
@pytest.mark.parametrize("a, b, expected", [(1,1,2), (2,2,4)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# ===== Mocking =====
from unittest.mock import patch, MagicMock
with patch("module.function_name") as mock_func:
    mock_func.return_value = "قيمة وهمية"

# ===== أوامر التشغيل =====
# uv run pytest                              تشغيل كل الاختبارات
# uv run pytest -v                           تفصيلي
# uv run pytest tests/test_x.py::test_y      اختبار محدد
# uv run pytest --cov=src                    مع تقرير التغطية
# uv run pytest -k "keyword"                 فلترة بالاسم
