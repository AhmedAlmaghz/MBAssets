#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-037-backup_file.py
"""

from pathlib import Path
from datetime import datetime
import shutil

def backup_file(source: Path, backup_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"الملف الأصلي غير موجود: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(source, backup_path)
    return backup_path

def list_backups(backup_dir: Path, original_name: str) -> list[Path]:
    if not backup_dir.exists():
        return []

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    matches = list(backup_dir.glob(f"{stem}_*{suffix}"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def restore_latest(backup_dir: Path, original_name: str, destination: Path) -> bool:
    backups = list_backups(backup_dir, original_name)
    if not backups:
        return False

    latest_backup = backups[0]
    shutil.copy2(latest_backup, destination)
    return True

# تجربة
source_file = Path("notes.txt")
source_file.write_text("محتوى تجريبي أول", encoding="utf-8")

backup_dir = Path("backups")

backup_file(source_file, backup_dir)

import time
time.sleep(1)  # للتأكد من اختلاف الطابع الزمني بين النسختين

source_file.write_text("محتوى تجريبي ثانٍ - مُحدّث", encoding="utf-8")
backup_file(source_file, backup_dir)

backups = list_backups(backup_dir, "notes.txt")
print(f"عدد النسخ الاحتياطية: {len(backups)}")
for b in backups:
    print(f"  {b.name}")

success = restore_latest(backup_dir, "notes.txt", Path("restored_notes.txt"))
print(f"تمت الاستعادة بنجاح: {success}")
print(Path("restored_notes.txt").read_text(encoding="utf-8"))
