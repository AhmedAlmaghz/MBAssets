#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-016-get_temperature.py
"""

# src/weather.py
import requests

def get_temperature(city: str) -> float:
    """جلب درجة الحرارة الحالية لمدينة عبر API خارجي."""
    response = requests.get(f"https://api.weather.example.com/{city}", timeout=10)
    data = response.json()
    return data["temperature"]
