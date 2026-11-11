#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-011-cli.py
"""

"""واجهة سطر الأوامر لأداة تنظيم الملفات."""

from pathlib import Path

import click

from file_organizer.organizer import execute_moves, scan_directory, summarize_by_category

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🗂️ أداة تنظيم الملفات تلقائيًا حسب نوعها."""
    pass

@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def preview(directory: Path):
    """معاينة خطة التنظيم دون تنفيذ أي تغيير فعلي (وضع آمن)."""
    moves = scan_directory(directory)

    if not moves:
        click.secho("📭 لا توجد ملفات لتنظيمها في هذا المجلد.", fg="yellow")
        return

    click.secho(f"📋 خطة تنظيم {len(moves)} ملف:\n", bold=True)

    summary = summarize_by_category(moves)
    for category, count in summary.items():
        click.echo(f"  📁 {category}: {count} ملف")

    click.echo()
    click.secho(
        "💡 هذه معاينة فقط. استخدم الأمر 'organize' لتنفيذ التغييرات فعليًا.",
        fg="cyan",
    )

@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.confirmation_option(
    prompt="⚠️ سيتم نقل الملفات فعليًا. هل أنت متأكد؟"
)
def organize(directory: Path):
    """تنظيم الملفات فعليًا في المجلد المحدد حسب نوعها."""
    moves = scan_directory(directory)

    if not moves:
        click.secho("📭 لا توجد ملفات لتنظيمها في هذا المجلد.", fg="yellow")
        return

    click.echo(f"🚀 جاري تنظيم {len(moves)} ملف...\n")

    with click.progressbar(moves, label="التقدّم") as bar:
        success_count, errors = execute_moves(list(bar))

    click.echo()
    click.secho(f"✅ تم نقل {success_count} ملف بنجاح!", fg="green", bold=True)

    if errors:
        click.secho(f"\n⚠️ حدثت {len(errors)} مشكلة:", fg="yellow")
        for error in errors:
            click.echo(f"  {error}")

@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def stats(directory: Path):
    """عرض إحصائيات عن أنواع الملفات الموجودة في المجلد."""
    moves = scan_directory(directory)
    summary = summarize_by_category(moves)

    if not summary:
        click.secho("📭 المجلد فارغ.", fg="yellow")
        return

    total = sum(summary.values())
    click.secho(f"📊 إحصائيات المجلد ({total} ملف إجمالاً):\n", bold=True)

    for category, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 5)
        click.echo(f"  {category:12s} {bar} {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    cli()
