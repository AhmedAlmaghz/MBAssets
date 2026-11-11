#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-010-temp_file.py
"""

import pytest
from pathlib import Path

@pytest.fixture
def temp_file(tmp_path):
    """إنشاء ملف مؤقت للاختبار، وحذفه تلقائيًا بعد الانتهاء."""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("بيانات تجريبية", encoding="utf-8")

    yield file_path  # الاختبار يستخدم هذه القيمة

    # هذا الجزء يُنفَّذ تلقائيًا بعد انتهاء الاختبار (تنظيف)
    print(f"تنظيف: {file_path}")

def test_file_content(temp_file):
    content = temp_file.read_text(encoding="utf-8")
    assert content == "بيانات تجريبية"
