#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-017-test_get_temperature_with_mock.py
"""

# tests/test_weather.py
from unittest.mock import MagicMock, patch

from weather import get_temperature

def test_get_temperature_with_mock():
    # استبدال requests.get بنسخة وهمية لا تتصل فعليًا بالإنترنت
    with patch("weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25.5}
        mock_get.return_value = mock_response

        result = get_temperature("الرياض")

        assert result == 25.5
        mock_get.assert_called_once()  # التحقق من أن الدالة استُدعيت فعلاً
