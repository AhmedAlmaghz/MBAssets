#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.9: متغيرات البيئة وإدارة الأسرار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-005-database_url.py
"""

from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ ملف .env ويضع قيمه في متغيرات البيئة

database_url = os.environ["DATABASE_URL"]
secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")
