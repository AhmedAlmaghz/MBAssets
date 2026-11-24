#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
Python 3.14 — From Zero to Pro — Packaging

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-26-chapter-26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
Python 3.14 — From Zero to Pro — Packaging

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["dotenv"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q python-dotenv

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# إعداد خاص بهذا الفصل
import os
os.environ.setdefault('DATABASE_URL', 'sqlite:///app.db')
os.environ.setdefault('SECRET_KEY', 'dev-secret')
pathlib.Path('.env').write_text(
    'DATABASE_URL=sqlite:///app.db\nSECRET_KEY=dev-secret\n', encoding='utf-8')

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""أدوات معالجة وتنسيق النصوص."""

import re

def slugify(text: str) -> str:
    """
    يحوّل نصًا إلى صيغة "slug" مناسبة للروابط.

    slugify("Hello World!")
    'hello-world'
    slugify("  Python 3.14  Guide  ")
    'python-3-14-guide'
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    يختصر نصًا طويلاً ويضيف لاحقة إن تجاوز الطول المحدد.

    truncate("Python is amazing", 10)
    'Python is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix

def word_count(text: str) -> int:
    """يُحصي عدد الكلمات في نص."""
    return len(text.split())

def is_palindrome(text: str) -> bool:
    """
    يتحقق إن كان النص متناظرًا (يُقرأ نفسه عكسيًا).

    is_palindrome("Was it a car or a cat I saw")
    True
    """
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned == cleaned[::-1]

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
## الاستخدام السريع

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ ملف .env ويضع قيمه في متغيرات البيئة

database_url = os.environ["DATABASE_URL"]
secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch26/26-001-slugify.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-001-slugify.py
"""

"""أدوات معالجة وتنسيق النصوص."""

import re

def slugify(text: str) -> str:
    """
    يحوّل نصًا إلى صيغة "slug" مناسبة للروابط.

    slugify("Hello World!")
    'hello-world'
    slugify("  Python 3.14  Guide  ")
    'python-3-14-guide'
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    يختصر نصًا طويلاً ويضيف لاحقة إن تجاوز الطول المحدد.

    truncate("Python is amazing", 10)
    'Python is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix

def word_count(text: str) -> int:
    """يُحصي عدد الكلمات في نص."""
    return len(text.split())

def is_palindrome(text: str) -> bool:
    """
    يتحقق إن كان النص متناظرًا (يُقرأ نفسه عكسيًا).

    is_palindrome("Was it a car or a cat I saw")
    True
    """
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned == cleaned[::-1]

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch26/26-002-__version__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-002-__version__.py
"""

"""textkit: مجموعة أدوات بسيطة لمعالجة النصوص."""

from textkit.formatters import is_palindrome, slugify, truncate, word_count

__version__ = "0.1.0"

__all__ = [
    "slugify",
    "truncate",
    "word_count",
    "is_palindrome",
]

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch26/26-003-test_simple_sentence.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-003-test_simple_sentence.py
"""

import pytest

from textkit import is_palindrome, slugify, truncate, word_count

class TestSlugify:
    def test_simple_sentence(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("  Python   Guide  ") == "python-guide"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

class TestTruncate:
    @pytest.mark.parametrize(
        "text, max_length, expected",
        [
            ("Python", 10, "Python"),
            ("Python is amazing", 10, "Python is..."),
            ("", 5, ""),
        ],
    )
    def test_truncate_cases(
        self, text: str, max_length: int, expected: str
    ) -> None:
        assert truncate(text, max_length) == expected

def test_word_count() -> None:
    assert word_count("Python is fun") == 3

def test_is_palindrome() -> None:
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("Hello") is False

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch26/26-004-ex26_7.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.7: ملفات لا غنى عنها في أي حزمة احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-004-ex26_7.py
"""

from textkit import slugify, truncate

slugify("Hello World!")       # "hello-world"
truncate("نص طويل جداً", 5)   # "نص طو..."

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch26/26-005-database_url.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.9: متغيرات البيئة وإدارة الأسرار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-005-database_url.py
"""

from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ ملف .env ويضع قيمه في متغيرات البيئة

database_url = os.environ["DATABASE_URL"]
secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch26/26-006-read_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-006-read_root.py
"""

"""واجهة API بسيطة تستخدم مكتبة textkit."""

from fastapi import FastAPI
from pydantic import BaseModel

from textkit import slugify, truncate

app = FastAPI(
    title="Textkit API",
    description="واجهة ويب لأدوات معالجة النصوص",
    version="0.1.0",
)

class TextRequest(BaseModel):
    text: str

class TruncateRequest(BaseModel):
    text: str
    max_length: int = 50

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في Textkit API"}

@app.post("/slugify")
def slugify_endpoint(request: TextRequest) -> dict[str, str]:
    return {"result": slugify(request.text)}

@app.post("/truncate")
def truncate_endpoint(request: TruncateRequest) -> dict[str, str]:
    return {"result": truncate(request.text, request.max_length)}

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch26/26-007-test_read_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-007-test_read_root.py
"""

from fastapi.testclient import TestClient

from textkit_api.main import app

client = TestClient(app)

def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_slugify_endpoint() -> None:
    response = client.post("/slugify", json={"text": "Hello World!"})
    assert response.status_code == 200
    assert response.json() == {"result": "hello-world"}

def test_truncate_endpoint() -> None:
    response = client.post(
        "/truncate", json={"text": "Python is amazing", "max_length": 10}
    )
    assert response.status_code == 200
    assert response.json() == {"result": "Python is..."}

# ────────────────────────────────────────────────────────────────────────
# [012] dist/markdown/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
"""textkit: مجموعة أدوات بسيطة لمعالجة النصوص."""

from textkit.formatters import is_palindrome, slugify, truncate, word_count

__version__ = "0.1.0"

__all__ = [
    "slugify",
    "truncate",
    "word_count",
    "is_palindrome",
]

# ────────────────────────────────────────────────────────────────────────
# [013] dist/markdown/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
import pytest

from textkit import is_palindrome, slugify, truncate, word_count

class TestSlugify:
    def test_simple_sentence(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("  Python   Guide  ") == "python-guide"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

class TestTruncate:
    @pytest.mark.parametrize(
        "text, max_length, expected",
        [
            ("Python", 10, "Python"),
            ("Python is amazing", 10, "Python is..."),
            ("", 5, ""),
        ],
    )
    def test_truncate_cases(
        self, text: str, max_length: int, expected: str
    ) -> None:
        assert truncate(text, max_length) == expected

def test_word_count() -> None:
    assert word_count("Python is fun") == 3

def test_is_palindrome() -> None:
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("Hello") is False

# ────────────────────────────────────────────────────────────────────────
# [014] dist/markdown/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from textkit import slugify, truncate

slugify("Hello World!")       # "hello-world"
truncate("نص طويل جداً", 5)   # "نص طو..."

# ────────────────────────────────────────────────────────────────────────
# [015] dist/markdown/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
> 💡 **نصيحة الخبير:** في مشاريع FastAPI الحديثة، يُفضّل استخدام `pydantic-settings` بدلاً من `os.environ` المباشر — فهي توفر Type Hints وتحقّقًا تلقائيًا من القيم:
> ```python
> from pydantic_settings import BaseSettings
>
> class Settings(BaseSettings):
>     database_url: str
>     secret_key: str
>     debug: bool = False
>
>     class Config:
>         env_file = ".env"
>
> settings = Settings()
> ```

> ⚠️ **خطأ شائع:** رفع ملف `.env` عن طريق الخطأ لأنك أضفته لـ `.gitignore` *بعد* أن كان قد تمّت إضافته مسبقًا لـ Git. تحقّق دائمًا بالأمر `git status` قبل أول `commit`، وإن حدث الخطأ، استخدم `git rm --cached .env` فورًا ثم غيّر كل المفاتيح المسرّبة فورًا (لا يكفي حذفها من الكود).

---

## 26.10 مقدمة إلى Docker: «يعمل على جهازي» تصبح من الماضي

### المشكلة التي يحلّها Docker

عبارة «يعمل عندي لكن لا يعمل على الخادم» هي من أشهر جمل الإحباط في تاريخ البرمجة. السبب: فروقات دقيقة بين البيئات — إصدار بايثون مختلف، مكتبة نظام مفقودة، متغير بيئة منسي.

**Docker** يحل هذا بفكرة بسيطة وعبقرية: بدلاً من نشر الكود فقط، ننشر **حاوية (Container)** تحتوي كل شيء يحتاجه التطبيق ليعمل — نظام التشغيل المصغّر، بايثون بالإصدار الصحيح، كل المكتبات، والكود نفسه — كوحدة واحدة متكاملة تعمل **بنفس الطريقة تمامًا** على أي جهاز: حاسوبك، خادم الشركة، أو سحابة Amazon.

> 🔄 **مقارنة: الحاويات (Containers) مقابل الآلات الافتراضية (Virtual Machines)**

| المعيار | Virtual Machine | Docker Container |
|---|---|---|
| تحتوي نظام تشغيل كامل؟ | نعم (ثقيلة، جيجابايتات) | لا، تشارك نواة النظام المضيف (خفيفة، ميجابايتات) |
| وقت الإقلاع | دقائق | ثوانٍ |
| الاستخدام الشائع | عزل كامل بين أنظمة مختلفة | تعبئة تطبيق واحد ببيئته الدقيقة |

### المفاهيم الأساسية

| المصطلح | الشرح |
|---|---|
| **Image (صورة)** | «القالب» الثابت الذي يحتوي كل ما يحتاجه التطبيق (كتوصيف وصفة طبخ) |
| **Container (حاوية)** | نسخة «حيّة» قيد التشغيل من الصورة (كالطبق المطبوخ فعليًا من الوصفة) |
| **Dockerfile** | ملف نصي يصف خطوة بخطوة كيف تُبنى الصورة |
| **Docker Hub** | مستودع عام لمشاركة الصور الجاهزة (مثل PyPI لكن للحاويات) |

### تثبيت Docker

نزّل **Docker Desktop** من [docker.com](https://www.docker.com/products/docker-desktop) (متوفر لـ Windows وMac وLinux). بعد التثبيت، تحقّق بالأمر:

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["dotenv"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q python-dotenv

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# إعداد خاص بهذا الفصل
import os
os.environ.setdefault('DATABASE_URL', 'sqlite:///app.db')
os.environ.setdefault('SECRET_KEY', 'dev-secret')
pathlib.Path('.env').write_text(
    'DATABASE_URL=sqlite:///app.db\nSECRET_KEY=dev-secret\n', encoding='utf-8')

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""أدوات معالجة وتنسيق النصوص."""

import re

def slugify(text: str) -> str:
    """
    يحوّل نصًا إلى صيغة "slug" مناسبة للروابط.

    slugify("Hello World!")
    'hello-world'
    slugify("  Python 3.14  Guide  ")
    'python-3-14-guide'
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    يختصر نصًا طويلاً ويضيف لاحقة إن تجاوز الطول المحدد.

    truncate("Python is amazing", 10)
    'Python is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix

def word_count(text: str) -> int:
    """يُحصي عدد الكلمات في نص."""
    return len(text.split())

def is_palindrome(text: str) -> bool:
    """
    يتحقق إن كان النص متناظرًا (يُقرأ نفسه عكسيًا).

    is_palindrome("Was it a car or a cat I saw")
    True
    """
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned == cleaned[::-1]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
## الاستخدام السريع

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ ملف .env ويضع قيمه في متغيرات البيئة

database_url = os.environ["DATABASE_URL"]
secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-001-slugify.py
"""

"""أدوات معالجة وتنسيق النصوص."""

import re

def slugify(text: str) -> str:
    """
    يحوّل نصًا إلى صيغة "slug" مناسبة للروابط.

    slugify("Hello World!")
    'hello-world'
    slugify("  Python 3.14  Guide  ")
    'python-3-14-guide'
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    يختصر نصًا طويلاً ويضيف لاحقة إن تجاوز الطول المحدد.

    truncate("Python is amazing", 10)
    'Python is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix

def word_count(text: str) -> int:
    """يُحصي عدد الكلمات في نص."""
    return len(text.split())

def is_palindrome(text: str) -> bool:
    """
    يتحقق إن كان النص متناظرًا (يُقرأ نفسه عكسيًا).

    is_palindrome("Was it a car or a cat I saw")
    True
    """
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned == cleaned[::-1]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-002-__version__.py
"""

"""textkit: مجموعة أدوات بسيطة لمعالجة النصوص."""

from textkit.formatters import is_palindrome, slugify, truncate, word_count

__version__ = "0.1.0"

__all__ = [
    "slugify",
    "truncate",
    "word_count",
    "is_palindrome",
]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-003-test_simple_sentence.py
"""

import pytest

from textkit import is_palindrome, slugify, truncate, word_count

class TestSlugify:
    def test_simple_sentence(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("  Python   Guide  ") == "python-guide"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

class TestTruncate:
    @pytest.mark.parametrize(
        "text, max_length, expected",
        [
            ("Python", 10, "Python"),
            ("Python is amazing", 10, "Python is..."),
            ("", 5, ""),
        ],
    )
    def test_truncate_cases(
        self, text: str, max_length: int, expected: str
    ) -> None:
        assert truncate(text, max_length) == expected

def test_word_count() -> None:
    assert word_count("Python is fun") == 3

def test_is_palindrome() -> None:
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("Hello") is False

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.7: ملفات لا غنى عنها في أي حزمة احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-004-ex26_7.py
"""

from textkit import slugify, truncate

slugify("Hello World!")       # "hello-world"
truncate("نص طويل جداً", 5)   # "نص طو..."

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.9: متغيرات البيئة وإدارة الأسرار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-005-database_url.py
"""

from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ ملف .env ويضع قيمه في متغيرات البيئة

database_url = os.environ["DATABASE_URL"]
secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-006-read_root.py
"""

"""واجهة API بسيطة تستخدم مكتبة textkit."""

from fastapi import FastAPI
from pydantic import BaseModel

from textkit import slugify, truncate

app = FastAPI(
    title="Textkit API",
    description="واجهة ويب لأدوات معالجة النصوص",
    version="0.1.0",
)

class TextRequest(BaseModel):
    text: str

class TruncateRequest(BaseModel):
    text: str
    max_length: int = 50

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في Textkit API"}

@app.post("/slugify")
def slugify_endpoint(request: TextRequest) -> dict[str, str]:
    return {"result": slugify(request.text)}

@app.post("/truncate")
def truncate_endpoint(request: TruncateRequest) -> dict[str, str]:
    return {"result": truncate(request.text, request.max_length)}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-007-test_read_root.py
"""

from fastapi.testclient import TestClient

from textkit_api.main import app

client = TestClient(app)

def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_slugify_endpoint() -> None:
    response = client.post("/slugify", json={"text": "Hello World!"})
    assert response.status_code == 200
    assert response.json() == {"result": "hello-world"}

def test_truncate_endpoint() -> None:
    response = client.post(
        "/truncate", json={"text": "Python is amazing", "max_length": 10}
    )
    assert response.status_code == 200
    assert response.json() == {"result": "Python is..."}

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
"""textkit: مجموعة أدوات بسيطة لمعالجة النصوص."""

from textkit.formatters import is_palindrome, slugify, truncate, word_count

__version__ = "0.1.0"

__all__ = [
    "slugify",
    "truncate",
    "word_count",
    "is_palindrome",
]

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import pytest

from textkit import is_palindrome, slugify, truncate, word_count

class TestSlugify:
    def test_simple_sentence(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("  Python   Guide  ") == "python-guide"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

class TestTruncate:
    @pytest.mark.parametrize(
        "text, max_length, expected",
        [
            ("Python", 10, "Python"),
            ("Python is amazing", 10, "Python is..."),
            ("", 5, ""),
        ],
    )
    def test_truncate_cases(
        self, text: str, max_length: int, expected: str
    ) -> None:
        assert truncate(text, max_length) == expected

def test_word_count() -> None:
    assert word_count("Python is fun") == 3

def test_is_palindrome() -> None:
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("Hello") is False

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from textkit import slugify, truncate

slugify("Hello World!")       # "hello-world"
truncate("نص طويل جداً", 5)   # "نص طو..."

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-26-chapter-26.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
> 💡 **نصيحة الخبير:** في مشاريع FastAPI الحديثة، يُفضّل استخدام `pydantic-settings` بدلاً من `os.environ` المباشر — فهي توفر Type Hints وتحقّقًا تلقائيًا من القيم:
> ```python
> from pydantic_settings import BaseSettings
>
> class Settings(BaseSettings):
>     database_url: str
>     secret_key: str
>     debug: bool = False
>
>     class Config:
>         env_file = ".env"
>
> settings = Settings()
> ```

> ⚠️ **خطأ شائع:** رفع ملف `.env` عن طريق الخطأ لأنك أضفته لـ `.gitignore` *بعد* أن كان قد تمّت إضافته مسبقًا لـ Git. تحقّق دائمًا بالأمر `git status` قبل أول `commit`، وإن حدث الخطأ، استخدم `git rm --cached .env` فورًا ثم غيّر كل المفاتيح المسرّبة فورًا (لا يكفي حذفها من الكود).

---

## 26.10 مقدمة إلى Docker: «يعمل على جهازي» تصبح من الماضي

### المشكلة التي يحلّها Docker

عبارة «يعمل عندي لكن لا يعمل على الخادم» هي من أشهر جمل الإحباط في تاريخ البرمجة. السبب: فروقات دقيقة بين البيئات — إصدار بايثون مختلف، مكتبة نظام مفقودة، متغير بيئة منسي.

**Docker** يحل هذا بفكرة بسيطة وعبقرية: بدلاً من نشر الكود فقط، ننشر **حاوية (Container)** تحتوي كل شيء يحتاجه التطبيق ليعمل — نظام التشغيل المصغّر، بايثون بالإصدار الصحيح، كل المكتبات، والكود نفسه — كوحدة واحدة متكاملة تعمل **بنفس الطريقة تمامًا** على أي جهاز: حاسوبك، خادم الشركة، أو سحابة Amazon.

> 🔄 **مقارنة: الحاويات (Containers) مقابل الآلات الافتراضية (Virtual Machines)**

| المعيار | Virtual Machine | Docker Container |
|---|---|---|
| تحتوي نظام تشغيل كامل؟ | نعم (ثقيلة، جيجابايتات) | لا، تشارك نواة النظام المضيف (خفيفة، ميجابايتات) |
| وقت الإقلاع | دقائق | ثوانٍ |
| الاستخدام الشائع | عزل كامل بين أنظمة مختلفة | تعبئة تطبيق واحد ببيئته الدقيقة |

### المفاهيم الأساسية

| المصطلح | الشرح |
|---|---|
| **Image (صورة)** | «القالب» الثابت الذي يحتوي كل ما يحتاجه التطبيق (كتوصيف وصفة طبخ) |
| **Container (حاوية)** | نسخة «حيّة» قيد التشغيل من الصورة (كالطبق المطبوخ فعليًا من الوصفة) |
| **Dockerfile** | ملف نصي يصف خطوة بخطوة كيف تُبنى الصورة |
| **Docker Hub** | مستودع عام لمشاركة الصور الجاهزة (مثل PyPI لكن للحاويات) |

### تثبيت Docker

نزّل **Docker Desktop** من [docker.com](https://www.docker.com/products/docker-desktop) (متوفر لـ Windows وMac وLinux). بعد التثبيت، تحقّق بالأمر:
