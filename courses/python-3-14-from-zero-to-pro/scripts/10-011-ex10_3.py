#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-011-ex10_3.py
"""

# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)
