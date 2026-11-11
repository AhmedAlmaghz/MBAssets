#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-015-hello.py
"""

import click

# ===== أمر بسيط =====
@click.command()
def hello():
    """وصف الأمر (يظهر في --help)."""
    click.echo("مرحبًا!")

# ===== Argument مقابل Option =====
@click.argument("name")                          # إلزامي، بدون --
@click.option("--greeting", default="مرحبًا")     # اختياري، بـ --
@click.option("--loud", is_flag=True)             # علم منطقي True/False
@click.option("--age", type=click.IntRange(0, 120))
@click.option("--level", type=click.Choice(["a", "b"]))

# ===== أوامر فرعية (مثل git) =====
@click.group()
def cli():
    pass

@cli.command()
def sub_command():
    pass

# ===== ميزات مفيدة =====
click.secho("نص ملوّن", fg="green", bold=True)
click.confirmation_option(prompt="متأكد؟")
click.progressbar(items, label="جاري العمل")
type=click.Path(exists=True, path_type=Path)

# ===== تحويلها لأمر نظام =====
# في pyproject.toml:
# [project.scripts]
# mytool = "my_package.cli:cli"
