#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-006-set_password.py
"""

import click

@click.command()
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def set_password(password: str):
    """ضبط كلمة مرور بأمان (مخفية عند الكتابة)."""
    click.echo("✅ تم ضبط كلمة المرور بنجاح!")

if __name__ == "__main__":
    set_password()
