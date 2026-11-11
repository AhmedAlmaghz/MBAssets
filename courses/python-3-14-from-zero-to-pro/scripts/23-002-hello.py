#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-002-hello.py
"""

# hello.py
import click

@click.command()
def hello():
    """أداة ترحيب بسيطة."""
    click.echo("مرحبًا بك في عالم CLI!")

if __name__ == "__main__":
    hello()
