#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-014-organize_downloads.py
"""

from pathlib import Path
import shutil

def organize_downloads(downloads_path: Path) -> None:
    """يُنظّم الملفات في مجلد حسب امتدادها، كل نوع في مجلد فرعي خاص به."""
    extension_folders: dict[str, str] = {
        ".pdf": "مستندات",
        ".jpg": "صور",
        ".png": "صور",
        ".mp4": "فيديوهات",
        ".zip": "أرشيف",
    }

    for file_path in downloads_path.iterdir():
        if file_path.is_file():
            folder_name = extension_folders.get(file_path.suffix.lower())
            if folder_name is not None:
                target_dir = downloads_path / folder_name
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"نُقل {file_path.name} إلى {folder_name}/")

organize_downloads(Path.home() / "Downloads")
