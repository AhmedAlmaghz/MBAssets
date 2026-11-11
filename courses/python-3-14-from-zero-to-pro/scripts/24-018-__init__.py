#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-018-__init__.py
"""

# ❌ مبالغة: نظام إعدادات معقّد لميزة بسيطة قد لا تُستخدم أبدًا
class ConfigurableGreeter:
    def __init__(self, language="ar", formal=False, emoji_support=True,
                 max_length=100, custom_templates=None, plugins=None):
        # ... 20 سطرًا من المرونة غير المستخدمة فعليًا

# ✅ ابدأ بسيطًا، وأضف المرونة فقط عند الحاجة الفعلية
def greet(name: str) -> str:
    return f"مرحبًا، {name}!"
