#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-003-greet.py
"""

import click

@click.command()
@click.argument("name")
def greet(name: str):
    """الترحيب باسم محدد."""
    click.echo(f"أهلاً، {name}!")

if __name__ == "__main__":
    greet()
