#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — إدارة المشاريع كمحترف
Python 3.14 — From Zero to Pro — Chapter 18 — Modern Python

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-18-chapter-18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — إدارة المشاريع كمحترف
Python 3.14 — From Zero to Pro — Chapter 18 — Modern Python

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-18-chapter-18 (lesson 18)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests

response = requests.get("https://api.github.com")
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# src/currency_converter/converter.py
"""وحدة تحويل العملات."""
from __future__ import annotations

EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
}

class UnsupportedCurrencyError(Exception):
    """تُرفع عندما تكون العملة المطلوبة غير مدعومة."""

def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """
    يحوّل مبلغًا من عملة إلى أخرى بناءً على أسعار الصرف الثابتة.

    Args:
        amount: المبلغ المراد تحويله.
        from_currency: رمز العملة الأصلية (مثل "USD").
        to_currency: رمز العملة الهدف (مثل "SAR").

    Returns:
        المبلغ المحوّل بالعملة الهدف.

    Raises:
        UnsupportedCurrencyError: إن كانت إحدى العملتين غير مدعومة.
    """
    if from_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {from_currency}")
    if to_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {to_currency}")

    amount_in_usd = amount / EXCHANGE_RATES[from_currency]
    return round(amount_in_usd * EXCHANGE_RATES[to_currency], 2)

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# تعديل converter.py - إضافة عملات جديدة
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
    "AED": 3.67,   # جديد
    "KWD": 0.31,   # جديد
}

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch18/18-001-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.2: الجزء الأول: إدارة البيئات والتبعيات مع uv

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-001-response.py
"""

# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests

response = requests.get("https://api.github.com")
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch18/18-002-ex18_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.3: الجزء الثاني: هيكلة مشروع احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-002-ex18_3.py
"""

# src/my_project/__init__.py

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch18/18-003-__version__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.3: الجزء الثاني: هيكلة مشروع احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-003-__version__.py
"""

# src/my_project/__init__.py
from my_project.models import User
from my_project.utils import format_date

__version__ = "0.1.0"

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch18/18-004-__version__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-004-__version__.py
"""

# src/currency_converter/__init__.py
from currency_converter.converter import convert

__version__ = "0.1.0"
__all__ = ["convert"]

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch18/18-005-convert.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-005-convert.py
"""

# src/currency_converter/converter.py
"""وحدة تحويل العملات."""
from __future__ import annotations

EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
}

class UnsupportedCurrencyError(Exception):
    """تُرفع عندما تكون العملة المطلوبة غير مدعومة."""

def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """
    يحوّل مبلغًا من عملة إلى أخرى بناءً على أسعار الصرف الثابتة.

    Args:
        amount: المبلغ المراد تحويله.
        from_currency: رمز العملة الأصلية (مثل "USD").
        to_currency: رمز العملة الهدف (مثل "SAR").

    Returns:
        المبلغ المحوّل بالعملة الهدف.

    Raises:
        UnsupportedCurrencyError: إن كانت إحدى العملتين غير مدعومة.
    """
    if from_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {from_currency}")
    if to_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {to_currency}")

    amount_in_usd = amount / EXCHANGE_RATES[from_currency]
    return round(amount_in_usd * EXCHANGE_RATES[to_currency], 2)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch18/18-006-main.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-006-main.py
"""

# src/currency_converter/main.py
"""نقطة الدخول الرئيسية للبرنامج."""
from currency_converter.converter import convert, UnsupportedCurrencyError

def main() -> None:
    try:
        result = convert(100, "USD", "SAR")
        print(f"100 USD = {result} SAR")
    except UnsupportedCurrencyError as e:
        print(f"خطأ: {e}")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch18/18-007-ex18_6.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-007-ex18_6.py
"""

# تعديل converter.py - إضافة عملات جديدة
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
    "AED": 3.67,   # جديد
    "KWD": 0.31,   # جديد
}

# ────────────────────────────────────────────────────────────────────────
# [012] dist/markdown/chapter-18.md
# ────────────────────────────────────────────────────────────────────────
هذا يسمح للمستخدمين باستيراد `from my_project import User` مباشرة، بدلًا من `from my_project.models import User` الأطول.

### ملف `README.md`: واجهة مشروعك الأولى

كل مشروع احترافي يبدأ بملف `README.md` يشرح:

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests

response = requests.get("https://api.github.com")
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# src/currency_converter/converter.py
"""وحدة تحويل العملات."""
from __future__ import annotations

EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
}

class UnsupportedCurrencyError(Exception):
    """تُرفع عندما تكون العملة المطلوبة غير مدعومة."""

def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """
    يحوّل مبلغًا من عملة إلى أخرى بناءً على أسعار الصرف الثابتة.

    Args:
        amount: المبلغ المراد تحويله.
        from_currency: رمز العملة الأصلية (مثل "USD").
        to_currency: رمز العملة الهدف (مثل "SAR").

    Returns:
        المبلغ المحوّل بالعملة الهدف.

    Raises:
        UnsupportedCurrencyError: إن كانت إحدى العملتين غير مدعومة.
    """
    if from_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {from_currency}")
    if to_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {to_currency}")

    amount_in_usd = amount / EXCHANGE_RATES[from_currency]
    return round(amount_in_usd * EXCHANGE_RATES[to_currency], 2)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# تعديل converter.py - إضافة عملات جديدة
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
    "AED": 3.67,   # جديد
    "KWD": 0.31,   # جديد
}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.2: الجزء الأول: إدارة البيئات والتبعيات مع uv

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-001-response.py
"""

# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests

response = requests.get("https://api.github.com")
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.3: الجزء الثاني: هيكلة مشروع احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-002-ex18_3.py
"""

# src/my_project/__init__.py

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.3: الجزء الثاني: هيكلة مشروع احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-003-__version__.py
"""

# src/my_project/__init__.py
from my_project.models import User
from my_project.utils import format_date

__version__ = "0.1.0"

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-004-__version__.py
"""

# src/currency_converter/__init__.py
from currency_converter.converter import convert

__version__ = "0.1.0"
__all__ = ["convert"]

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-005-convert.py
"""

# src/currency_converter/converter.py
"""وحدة تحويل العملات."""
from __future__ import annotations

EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
}

class UnsupportedCurrencyError(Exception):
    """تُرفع عندما تكون العملة المطلوبة غير مدعومة."""

def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """
    يحوّل مبلغًا من عملة إلى أخرى بناءً على أسعار الصرف الثابتة.

    Args:
        amount: المبلغ المراد تحويله.
        from_currency: رمز العملة الأصلية (مثل "USD").
        to_currency: رمز العملة الهدف (مثل "SAR").

    Returns:
        المبلغ المحوّل بالعملة الهدف.

    Raises:
        UnsupportedCurrencyError: إن كانت إحدى العملتين غير مدعومة.
    """
    if from_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {from_currency}")
    if to_currency not in EXCHANGE_RATES:
        raise UnsupportedCurrencyError(f"العملة غير مدعومة: {to_currency}")

    amount_in_usd = amount / EXCHANGE_RATES[from_currency]
    return round(amount_in_usd * EXCHANGE_RATES[to_currency], 2)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-006-main.py
"""

# src/currency_converter/main.py
"""نقطة الدخول الرئيسية للبرنامج."""
from currency_converter.converter import convert, UnsupportedCurrencyError

def main() -> None:
    try:
        result = convert(100, "USD", "SAR")
        print(f"100 USD = {result} SAR")
    except UnsupportedCurrencyError as e:
        print(f"خطأ: {e}")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-007-ex18_6.py
"""

# تعديل converter.py - إضافة عملات جديدة
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "SAR": 3.75,
    "EGP": 48.5,
    "AED": 3.67,   # جديد
    "KWD": 0.31,   # جديد
}

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-18-chapter-18.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
هذا يسمح للمستخدمين باستيراد `from my_project import User` مباشرة، بدلًا من `from my_project.models import User` الأطول.

### ملف `README.md`: واجهة مشروعك الأولى

كل مشروع احترافي يبدأ بملف `README.md` يشرح:
