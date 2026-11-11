#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.4: بناء أوامر فرعية متعددة: `click.group()`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-008-cli.py
"""

# cli.py
import click

@click.group()
def cli():
    """أداة إدارة المهام من سطر الأوامر."""
    pass

@cli.command()
@click.argument("task")
def add(task: str):
    """إضافة مهمة جديدة."""
    click.echo(f"✅ تمت إضافة المهمة: {task}")

@cli.command()
def list():
    """عرض كل المهام."""
    click.echo("📋 قائمة المهام:")
    click.echo("1. تعلّم click")
    click.echo("2. بناء مشروع CLI")

@cli.command()
@click.argument("task_id", type=int)
def done(task_id: int):
    """تحديد مهمة كمنجزة."""
    click.echo(f"🎉 تم إنجاز المهمة رقم {task_id}")

if __name__ == "__main__":
    cli()
