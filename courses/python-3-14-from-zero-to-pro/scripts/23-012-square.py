#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-012-square.py
"""

import click

# 1
@click.command()
@click.argument("number", type=int)
def square(number: int):
    click.echo(f"{number}² = {number ** 2}")

# 2
@click.command()
@click.argument("value", type=float)
@click.option("--unit", type=click.Choice(["celsius", "fahrenheit"]), default="celsius")
def convert_temp(value: float, unit: str):
    if unit == "celsius":
        result = value * 9 / 5 + 32
        click.echo(f"{value}°C = {result}°F")
    else:
        result = (value - 32) * 5 / 9
        click.echo(f"{value}°F = {result}°C")

# 3
@click.command()
@click.confirmation_option(prompt="هل تريد إعادة الضبط؟")
def reset():
    click.echo("تم إعادة الضبط!")

if __name__ == "__main__":
    square()
