# -*- coding: utf-8 -*-
"""
توافق خلفي (Backport) لمكتبة `annotationlib` القياسية في Python 3.14.

الكتاب يستهدف Python 3.14+ حيث تُقيَّم التعليقات النوعية كسولاً (PEP 649/749).
هذا الملف يوفّر واجهة مطابقة مبسّطة تعمل على Python 3.12/3.13 كي تبقى أمثلة
الفصلين الثاني والرابع قابلة للتشغيل على Colab وKaggle وأي بيئة أقدم.
"""
from __future__ import annotations

import sys
import typing

__all__ = ["Format", "get_annotations"]


class Format:
    """تنسيقات القراءة الثلاثة (مطابقة لأسماء Python 3.14)."""

    VALUE = 1
    FORWARDREF = 2
    STRING = 3


def get_annotations(
    obj: object,
    *,
    format: int = Format.VALUE,  # noqa: A002 - مطابقة لتوقيع المكتبة القياسية
    globals: dict | None = None,  # noqa: A002
    locals: dict | None = None,  # noqa: A002
    eval_str: bool = False,
) -> dict[str, typing.Any]:
    """يعيد تعليقات الكائن النوعية؛ يحلّ الإشارات الأمامية عند طلب VALUE."""
    if format == Format.STRING and not eval_str:
        return dict(getattr(obj, "__annotations__", {}) or {})

    try:
        return typing.get_type_hints(
            obj, globalns=globals, localns=locals, include_extras=True
        )
    except Exception:  # pragma: no cover - سلوك احتياطي متسامح
        return dict(getattr(obj, "__annotations__", {}) or {})


if sys.version_info >= (3, 14):  # pragma: no cover
    from annotationlib import Format, get_annotations  # type: ignore[no-redef]
