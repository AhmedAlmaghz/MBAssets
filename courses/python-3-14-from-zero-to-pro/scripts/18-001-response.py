#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.2: الجزء الأول: إدارة البيئات والتبعيات مع uv

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-001-response.py
"""

# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests

response = requests.get("https://api.github.com")
print(response.status_code)
