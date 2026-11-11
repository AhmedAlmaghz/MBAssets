#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-026-ex7_13.py
"""

# ❌ غامض عند التداخل
for i in range(10):
    for j in range(10):
        pass

# ✅ أوضح عند وجود معنى حقيقي
for row in range(10):
    for column in range(10):
        pass
