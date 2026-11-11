#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-008-ex11_3.py
"""

# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!
