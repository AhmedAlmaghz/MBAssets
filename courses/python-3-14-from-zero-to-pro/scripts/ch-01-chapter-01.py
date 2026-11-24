#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كيف يفكّر المبرمج؟
Python 3.14 — From Zero to Pro — Chapter 01 — Before the Code

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-01-chapter-01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كيف يفكّر المبرمج؟
Python 3.14 — From Zero to Pro — Chapter 01 — Before the Code

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-01-chapter-01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كيف يفكّر المبرمج؟
Python 3.14 — From Zero to Pro — Chapter 01 — Before the Code

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch01/01-001-is_leap_year.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الأول: كيف يفكّر المبرمج؟
القسم 1.6: عقلية حل المشكلات: المهارة الحقيقية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 01-001-is_leap_year.py
"""

def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الأول: كيف يفكّر المبرمج؟
القسم 1.6: عقلية حل المشكلات: المهارة الحقيقية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 01-001-is_leap_year.py
"""

def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كيف يفكّر المبرمج؟
Python 3.14 — From Zero to Pro — Chapter 01 — Before the Code

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-01-chapter-01 (lesson 1)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch01/01-001-is_leap_year.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الأول: كيف يفكّر المبرمج؟
القسم 1.6: عقلية حل المشكلات: المهارة الحقيقية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 01-001-is_leap_year.py
"""

def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-01-chapter-01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الأول: كيف يفكّر المبرمج؟
القسم 1.6: عقلية حل المشكلات: المهارة الحقيقية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 01-001-is_leap_year.py
"""

def is_leap_year(year: int) -> bool:
    """تحديد ما إذا كانت السنة كبيسة أم لا."""
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# اختبار الدالة
print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True
