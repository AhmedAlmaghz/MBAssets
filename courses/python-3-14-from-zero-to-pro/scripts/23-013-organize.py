#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-013-organize.py
"""

# تمرين 5 -- دمج المعاينة كخيار بدلاً من أمر منفصل
@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="معاينة فقط دون تنفيذ فعلي")
def organize(directory: Path, dry_run: bool):
    """تنظيم الملفات، مع إمكانية المعاينة عبر --dry-run."""
    moves = scan_directory(directory)

    if not moves:
        click.secho("📭 لا توجد ملفات لتنظيمها.", fg="yellow")
        return

    if dry_run:
        click.secho("🔍 وضع المعاينة (dry-run) -- لن يتم تنفيذ أي تغيير:\n", fg="cyan")
        summary = summarize_by_category(moves)
        for category, count in summary.items():
            click.echo(f"  📁 {category}: {count} ملف")
        return

    # التنفيذ الفعلي كما في الكود الأصلي...
    success_count, errors = execute_moves(moves)
    click.secho(f"✅ تم نقل {success_count} ملف!", fg="green")
