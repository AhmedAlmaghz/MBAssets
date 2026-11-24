#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار بناء أدوات سطر الأوامر (CLI)
Python 3.14 — From Zero to Pro — CLI

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 27 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-23-chapter-23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار بناء أدوات سطر الأوامر (CLI)
Python 3.14 — From Zero to Pro — CLI

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 26 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["click"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q click

# إعداد خاص بهذا الفصل
# الدفتر ليس طرفية: نعطّل كتل __main__ حتى لا تخرج الأدوات بـ SystemExit.
__name__ = 'notebook'

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# hello.py
import click

@click.command()
def hello():
    """أداة ترحيب بسيطة."""
    click.echo("مرحبًا بك في عالم CLI!")

if __name__ == "__main__":
    hello()

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.argument("name")
def greet(name: str):
    """الترحيب باسم محدد."""
    click.echo(f"أهلاً، {name}!")

if __name__ == "__main__":
    greet()

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def set_password(password: str):
    """ضبط كلمة مرور بأمان (مخفية عند الكتابة)."""
    click.echo("✅ تم ضبط كلمة المرور بنجاح!")

if __name__ == "__main__":
    set_password()

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-23.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-23.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch23/23-001-parser.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.2: البداية البدائية: وحدة `argparse` المدمجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-001-parser.py
"""

# greet.py
import argparse

parser = argparse.ArgumentParser(description="أداة ترحيب بسيطة")
parser.add_argument("name", help="اسم الشخص المراد الترحيب به")
parser.add_argument("--greeting", default="مرحبًا", help="كلمة الترحيب")
parser.add_argument("--loud", action="store_true", help="اطبع بأحرف كبيرة")

args = parser.parse_args()

message = f"{args.greeting}، {args.name}!"
if args.loud:
    message = message.upper()

print(message)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch23/23-002-hello.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch23/23-003-greet.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] code/chapters/ch23/23-004-greet.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] code/chapters/ch23/23-005-register.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] code/chapters/ch23/23-006-set_password.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] code/chapters/ch23/23-007-ex23_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-007-ex23_3.py
"""

import click

click.secho("نجاح! ✅", fg="green", bold=True)
click.secho("تحذير! ⚠️", fg="yellow")
click.secho("خطأ! ❌", fg="red", bold=True)

# ────────────────────────────────────────────────────────────────────────
# [016] code/chapters/ch23/23-008-cli.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch23/23-009-get_category_for_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-009-get_category_for_file.py
"""

"""قواعد تصنيف أنواع الملفات إلى فئات."""

FILE_CATEGORIES: dict[str, list[str]] = {
    "الصور": [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"],
    "المستندات": [".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx"],
    "الفيديوهات": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "الصوتيات": [".mp3", ".wav", ".flac", ".aac"],
    "الأرشيف": [".zip", ".rar", ".tar", ".gz", ".7z"],
    "الأكواد": [".py", ".js", ".html", ".css", ".json", ".java", ".cpp"],
}

DEFAULT_CATEGORY = "أخرى"

def get_category_for_file(extension: str) -> str:
    """تحديد الفئة المناسبة لامتداد ملف معيّن.

    Args:
        extension: امتداد الملف (مثل '.jpg'), بأي حالة أحرف.

    Returns:
        اسم الفئة المناسبة، أو DEFAULT_CATEGORY إن لم يُعرف الامتداد.
    """
    extension = extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return DEFAULT_CATEGORY

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch23/23-010-scan_directory.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-010-scan_directory.py
"""

"""منطق فحص وتنظيم الملفات في مجلد."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from file_organizer.rules import get_category_for_file

@dataclass
class FileMove:
    """يمثّل عملية نقل ملف واحد مخطَّط لها."""

    source: Path
    destination: Path
    category: str

def scan_directory(directory: Path) -> list[FileMove]:
    """فحص مجلد وتحديد وجهة كل ملف فيه دون تنفيذ أي نقل فعلي.

    Args:
        directory: المجلد المراد فحصه.

    Returns:
        قائمة بعمليات النقل المخطَّط لها (FileMove) لكل ملف.
    """
    planned_moves: list[FileMove] = []

    for item in directory.iterdir():
        if item.is_dir():
            continue  # نتجاهل المجلدات الفرعية الموجودة مسبقًا

        category = get_category_for_file(item.suffix)
        destination_folder = directory / category
        destination_path = destination_folder / item.name

        planned_moves.append(
            FileMove(source=item, destination=destination_path, category=category)
        )

    return planned_moves

def execute_moves(moves: list[FileMove]) -> tuple[int, list[str]]:
    """تنفيذ عمليات النقل المخطَّط لها فعليًا.

    Args:
        moves: قائمة عمليات النقل المطلوب تنفيذها.

    Returns:
        عدد العمليات الناجحة، وقائمة برسائل الأخطاء إن وُجدت.
    """
    success_count = 0
    errors: list[str] = []

    for move in moves:
        try:
            move.destination.parent.mkdir(exist_ok=True)

            if move.destination.exists():
                # تجنّب الكتابة فوق ملف موجود بنفس الاسم
                errors.append(f"⚠️ تخطّي {move.source.name}: يوجد ملف بنفس الاسم في الوجهة")
                continue

            shutil.move(str(move.source), str(move.destination))
            success_count += 1

        except OSError as e:
            errors.append(f"❌ فشل نقل {move.source.name}: {e}")

    return success_count, errors

def summarize_by_category(moves: list[FileMove]) -> dict[str, int]:
    """تلخيص عدد الملفات لكل فئة."""
    summary: dict[str, int] = {}
    for move in moves:
        summary[move.category] = summary.get(move.category, 0) + 1
    return summary

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch23/23-011-cli.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch23/23-012-square.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch23/23-013-organize.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch23/23-014-load_custom_rules.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-014-load_custom_rules.py
"""

import json
from pathlib import Path

from file_organizer.rules import FILE_CATEGORIES

def load_custom_rules() -> dict[str, list[str]]:
    """تحميل قواعد تصنيف مخصصة من ملف إعدادات إن وُجد، ودمجها مع الافتراضية."""
    config_locations = [Path.cwd() / ".forganizerc", Path.home() / ".forganizerc"]

    merged_rules = dict(FILE_CATEGORIES)  # نسخة من القواعد الافتراضية

    for config_path in config_locations:
        if config_path.exists():
            custom_rules = json.loads(config_path.read_text(encoding="utf-8"))
            merged_rules.update(custom_rules)  # القواعد المخصصة تطغى على الافتراضية
            break

    return merged_rules

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch23/23-015-hello.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [024] dist/markdown/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# greet.py
import argparse

parser = argparse.ArgumentParser(description="أداة ترحيب بسيطة")
parser.add_argument("name", help="اسم الشخص المراد الترحيب به")
parser.add_argument("--greeting", default="مرحبًا", help="كلمة الترحيب")
parser.add_argument("--loud", action="store_true", help="اطبع بأحرف كبيرة")

args = parser.parse_args()

message = f"{args.greeting}، {args.name}!"
if args.loud:
    message = message.upper()

print(message)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/markdown/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
import click

click.secho("نجاح! ✅", fg="green", bold=True)
click.secho("تحذير! ⚠️", fg="yellow")
click.secho("خطأ! ❌", fg="red", bold=True)

# ────────────────────────────────────────────────────────────────────────
# [026] dist/markdown/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
> import click
> from pathlib import Path
>
> @click.command()
> @click.argument("file_path", type=click.Path(exists=True, path_type=Path))
> def process(file_path: Path):
>     ""«معالجة ملف -- click يتحقق من وجوده تلقائيًا!»""
>     click.echo(f«معالجة: {file_path}»)
> ```
> لاحظ `type=click.Path(exists=True)` — يتحقق click تلقائيًا من وجود الملف، ويرفض الاستمرار برسالة خطأ واضحة إن لم يكن موجودًا، بدلاً من أن ينهار برنامجك لاحقًا بخطأ غامض.

> **⚠️ تحذير: عمليات خطيرة بدون تأكيد**
> أي أمر قد يحذف أو يُعدّل بيانات بشكل لا رجعة فيه يجب أن يطلب تأكيدًا:
> ```python
> @click.command()
> @click.confirmation_option(prompt=«هل أنت متأكد من حذف كل الملفات؟»)
> def delete_all():
>     click.echo(«تم الحذف.»)
> ```
> ```bash
> $ python cli.py delete_all
> هل أنت متأكد من حذف كل الملفات؟ [y/N]: n
> Aborted!
> ```

> **💡 نصيحة خبير: استخدام `click.progressbar` للعمليات الطويلة**
> ```python
> import time
> import click
>
> items = range(100)
> with click.progressbar(items, label=«معالجة الملفات») as bar:
>     for item in bar:
>         time.sleep(0.01)  # محاكاة عمل فعلي
> ```
> يُعطي المستخدم تغذية راجعة بصرية بدلاً من ترك البرنامج «صامتًا» لعدة ثوانٍ أو دقائق، مما يُحسّن تجربة الاستخدام بشكل كبير.

---

## 23.7 مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

سنبني أداة CLI احترافية حقيقية: تفحص مجلدًا (مثل مجلد التنزيلات)، وتُنظّم ملفاته تلقائيًا في مجلدات فرعية حسب نوعها (صور، مستندات، فيديوهات...)، مع خيارات متعددة ووضع «تجربة» آمن قبل التنفيذ الفعلي.

### هيكلية المشروع

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["click"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q click

# إعداد خاص بهذا الفصل
# الدفتر ليس طرفية: نعطّل كتل __main__ حتى لا تخرج الأدوات بـ SystemExit.
__name__ = 'notebook'

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# hello.py
import click

@click.command()
def hello():
    """أداة ترحيب بسيطة."""
    click.echo("مرحبًا بك في عالم CLI!")

if __name__ == "__main__":
    hello()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.argument("name")
def greet(name: str):
    """الترحيب باسم محدد."""
    click.echo(f"أهلاً، {name}!")

if __name__ == "__main__":
    greet()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def set_password(password: str):
    """ضبط كلمة مرور بأمان (مخفية عند الكتابة)."""
    click.echo("✅ تم ضبط كلمة المرور بنجاح!")

if __name__ == "__main__":
    set_password()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.2: البداية البدائية: وحدة `argparse` المدمجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-001-parser.py
"""

# greet.py
import argparse

parser = argparse.ArgumentParser(description="أداة ترحيب بسيطة")
parser.add_argument("name", help="اسم الشخص المراد الترحيب به")
parser.add_argument("--greeting", default="مرحبًا", help="كلمة الترحيب")
parser.add_argument("--loud", action="store_true", help="اطبع بأحرف كبيرة")

args = parser.parse_args()

message = f"{args.greeting}، {args.name}!"
if args.loud:
    message = message.upper()

print(message)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.3: مكتبة `click`: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-007-ex23_3.py
"""

import click

click.secho("نجاح! ✅", fg="green", bold=True)
click.secho("تحذير! ⚠️", fg="yellow")
click.secho("خطأ! ❌", fg="red", bold=True)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-009-get_category_for_file.py
"""

"""قواعد تصنيف أنواع الملفات إلى فئات."""

FILE_CATEGORIES: dict[str, list[str]] = {
    "الصور": [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"],
    "المستندات": [".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx"],
    "الفيديوهات": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "الصوتيات": [".mp3", ".wav", ".flac", ".aac"],
    "الأرشيف": [".zip", ".rar", ".tar", ".gz", ".7z"],
    "الأكواد": [".py", ".js", ".html", ".css", ".json", ".java", ".cpp"],
}

DEFAULT_CATEGORY = "أخرى"

def get_category_for_file(extension: str) -> str:
    """تحديد الفئة المناسبة لامتداد ملف معيّن.

    Args:
        extension: امتداد الملف (مثل '.jpg'), بأي حالة أحرف.

    Returns:
        اسم الفئة المناسبة، أو DEFAULT_CATEGORY إن لم يُعرف الامتداد.
    """
    extension = extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return DEFAULT_CATEGORY

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-010-scan_directory.py
"""

"""منطق فحص وتنظيم الملفات في مجلد."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from file_organizer.rules import get_category_for_file

@dataclass
class FileMove:
    """يمثّل عملية نقل ملف واحد مخطَّط لها."""

    source: Path
    destination: Path
    category: str

def scan_directory(directory: Path) -> list[FileMove]:
    """فحص مجلد وتحديد وجهة كل ملف فيه دون تنفيذ أي نقل فعلي.

    Args:
        directory: المجلد المراد فحصه.

    Returns:
        قائمة بعمليات النقل المخطَّط لها (FileMove) لكل ملف.
    """
    planned_moves: list[FileMove] = []

    for item in directory.iterdir():
        if item.is_dir():
            continue  # نتجاهل المجلدات الفرعية الموجودة مسبقًا

        category = get_category_for_file(item.suffix)
        destination_folder = directory / category
        destination_path = destination_folder / item.name

        planned_moves.append(
            FileMove(source=item, destination=destination_path, category=category)
        )

    return planned_moves

def execute_moves(moves: list[FileMove]) -> tuple[int, list[str]]:
    """تنفيذ عمليات النقل المخطَّط لها فعليًا.

    Args:
        moves: قائمة عمليات النقل المطلوب تنفيذها.

    Returns:
        عدد العمليات الناجحة، وقائمة برسائل الأخطاء إن وُجدت.
    """
    success_count = 0
    errors: list[str] = []

    for move in moves:
        try:
            move.destination.parent.mkdir(exist_ok=True)

            if move.destination.exists():
                # تجنّب الكتابة فوق ملف موجود بنفس الاسم
                errors.append(f"⚠️ تخطّي {move.source.name}: يوجد ملف بنفس الاسم في الوجهة")
                continue

            shutil.move(str(move.source), str(move.destination))
            success_count += 1

        except OSError as e:
            errors.append(f"❌ فشل نقل {move.source.name}: {e}")

    return success_count, errors

def summarize_by_category(moves: list[FileMove]) -> dict[str, int]:
    """تلخيص عدد الملفات لكل فئة."""
    summary: dict[str, int] = {}
    for move in moves:
        summary[move.category] = summary.get(move.category, 0) + 1
    return summary

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-014-load_custom_rules.py
"""

import json
from pathlib import Path

from file_organizer.rules import FILE_CATEGORIES

def load_custom_rules() -> dict[str, list[str]]:
    """تحميل قواعد تصنيف مخصصة من ملف إعدادات إن وُجد، ودمجها مع الافتراضية."""
    config_locations = [Path.cwd() / ".forganizerc", Path.home() / ".forganizerc"]

    merged_rules = dict(FILE_CATEGORIES)  # نسخة من القواعد الافتراضية

    for config_path in config_locations:
        if config_path.exists():
            custom_rules = json.loads(config_path.read_text(encoding="utf-8"))
            merged_rules.update(custom_rules)  # القواعد المخصصة تطغى على الافتراضية
            break

    return merged_rules

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# greet.py
import argparse

parser = argparse.ArgumentParser(description="أداة ترحيب بسيطة")
parser.add_argument("name", help="اسم الشخص المراد الترحيب به")
parser.add_argument("--greeting", default="مرحبًا", help="كلمة الترحيب")
parser.add_argument("--loud", action="store_true", help="اطبع بأحرف كبيرة")

args = parser.parse_args()

message = f"{args.greeting}، {args.name}!"
if args.loud:
    message = message.upper()

print(message)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import click

click.secho("نجاح! ✅", fg="green", bold=True)
click.secho("تحذير! ⚠️", fg="yellow")
click.secho("خطأ! ❌", fg="red", bold=True)

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-23-chapter-23.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
> import click
> from pathlib import Path
>
> @click.command()
> @click.argument("file_path", type=click.Path(exists=True, path_type=Path))
> def process(file_path: Path):
>     ""«معالجة ملف -- click يتحقق من وجوده تلقائيًا!»""
>     click.echo(f«معالجة: {file_path}»)
> ```
> لاحظ `type=click.Path(exists=True)` — يتحقق click تلقائيًا من وجود الملف، ويرفض الاستمرار برسالة خطأ واضحة إن لم يكن موجودًا، بدلاً من أن ينهار برنامجك لاحقًا بخطأ غامض.

> **⚠️ تحذير: عمليات خطيرة بدون تأكيد**
> أي أمر قد يحذف أو يُعدّل بيانات بشكل لا رجعة فيه يجب أن يطلب تأكيدًا:
> ```python
> @click.command()
> @click.confirmation_option(prompt=«هل أنت متأكد من حذف كل الملفات؟»)
> def delete_all():
>     click.echo(«تم الحذف.»)
> ```
> ```bash
> $ python cli.py delete_all
> هل أنت متأكد من حذف كل الملفات؟ [y/N]: n
> Aborted!
> ```

> **💡 نصيحة خبير: استخدام `click.progressbar` للعمليات الطويلة**
> ```python
> import time
> import click
>
> items = range(100)
> with click.progressbar(items, label=«معالجة الملفات») as bar:
>     for item in bar:
>         time.sleep(0.01)  # محاكاة عمل فعلي
> ```
> يُعطي المستخدم تغذية راجعة بصرية بدلاً من ترك البرنامج «صامتًا» لعدة ثوانٍ أو دقائق، مما يُحسّن تجربة الاستخدام بشكل كبير.

---

## 23.7 مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

سنبني أداة CLI احترافية حقيقية: تفحص مجلدًا (مثل مجلد التنزيلات)، وتُنظّم ملفاته تلقائيًا في مجلدات فرعية حسب نوعها (صور، مستندات، فيديوهات...)، مع خيارات متعددة ووضع «تجربة» آمن قبل التنفيذ الفعلي.

### هيكلية المشروع
