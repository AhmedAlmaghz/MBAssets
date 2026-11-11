#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-005-register.py
"""

import click

@click.command()
@click.option("--age", type=click.IntRange(0, 120), prompt=True, help="العمر")
@click.option(
    "--level",
    type=click.Choice(["مبتدئ", "متوسط", "متقدم"], case_sensitive=False),
    default="مبتدئ",
)
def register(age: int, level: str):
    """تسجيل مستخدم جديد مع تحقق مدمج من المدخلات."""
    click.echo(f"العمر: {age}, المستوى: {level}")

if __name__ == "__main__":
    register()
