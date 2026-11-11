#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-004-greet.py
"""

import click

@click.command()
@click.argument("name")
@click.option("--greeting", default="مرحبًا", help="كلمة الترحيب المستخدمة")
@click.option("--loud", is_flag=True, help="اطبع الرسالة بأحرف كبيرة")
@click.option("--times", default=1, type=int, help="عدد مرات تكرار الرسالة")
def greet(name: str, greeting: str, loud: bool, times: int):
    """أداة ترحيب قابلة للتخصيص."""
    message = f"{greeting}، {name}!"
    if loud:
        message = message.upper()

    for _ in range(times):
        click.echo(message)

if __name__ == "__main__":
    greet()
