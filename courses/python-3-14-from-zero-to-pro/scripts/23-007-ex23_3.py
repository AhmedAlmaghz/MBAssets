#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-007-ex23_3.py
"""

import click

click.secho("نجاح! ✅", fg="green", bold=True)
click.secho("تحذير! ⚠️", fg="yellow")
click.secho("خطأ! ❌", fg="red", bold=True)
