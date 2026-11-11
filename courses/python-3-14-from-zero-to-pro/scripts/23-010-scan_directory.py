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
