#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-009-fresh_data.py
"""

@pytest.fixture(scope="function")  # الافتراضي: يُعاد إنشاؤها لكل اختبار
def fresh_data():
    return []

@pytest.fixture(scope="module")  # تُنشأ مرة واحدة فقط لكل ملف اختبار
def database_connection():
    print("🔌 فتح اتصال بقاعدة البيانات...")
    connection = "وهمي-اتصال"
    yield connection  # القيمة التي تُمرَّر للاختبارات
    print("🔌 إغلاق الاتصال...")  # كود التنظيف، يُنفَّذ بعد انتهاء كل الاختبارات
