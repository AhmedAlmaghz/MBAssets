# -*- coding: utf-8 -*-
"""
from pathlib import Path

مقتطف من الفصل 8 — «بيئة التنفيذ المعزولة وأمن تشغيل الأكواد».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch08_sandbox_security.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from pathlib import Path


def extract_safe_artifacts(
    sandbox_dir: Path,
    *,
    excluded_filenames: set[str],
    allowed_extensions: frozenset[str],
    max_bytes_per_file: int = 500_000,
) -> dict[str, bytes]:
    """استخراج الملفات التي ولدها كود الوكيل داخل صندوق الرمل بأمان عسكري صارم"""
    safe_artifacts: dict[str, bytes] = {}
    resolved_root = sandbox_dir.resolve()

    for item in sandbox_dir.iterdir():
        if item.name in excluded_filenames:
            continue

        # 1. حظر الروابط الرمزية فوراً قبل محاولة قراءتها (Anti-Symlink Exfiltration)
        if item.is_symlink() or not item.is_file():
            continue

        # 2. التأكد الرياضي من أن المسار الحقيقي يقع داخل مجلد العزل (Anti-Path Traversal)
        resolved_item = item.resolve()
        if not resolved_item.is_relative_to(resolved_root):
            continue

        # 3. التحقق من القائمة البيضاء للامتدادات المسموحة
        if resolved_item.suffix.lower() not in allowed_extensions:
            continue

        # 4. فحص حجم الملف قبل قراءته في الذاكرة (Anti-Memory Bomb)
        file_size = resolved_item.stat().st_size
        if 0 < file_size <= max_bytes_per_file:
            safe_artifacts[item.name] = resolved_item.read_bytes()

    return safe_artifacts
