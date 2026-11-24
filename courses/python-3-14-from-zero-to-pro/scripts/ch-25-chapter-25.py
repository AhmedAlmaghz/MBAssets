#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الاختبار (Testing) باستخدام pytest
Python 3.14 — From Zero to Pro — Testing

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 53 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-25-chapter-25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الاختبار (Testing) باستخدام pytest
Python 3.14 — From Zero to Pro — Testing

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 52 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["pytest"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q pytest

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def test_addition():
    # Arrange (التحضير)
    a = 5
    b = 3

    # Act (التنفيذ)
    result = a + b

    # Assert (التحقق)
    assert result == 8

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# src/calculator.py
def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def divide(a: float, b: float) -> float:
    """قسمة رقمين، مع رفع استثناء عند القسمة على صفر."""
    if b == 0:
        raise ValueError("لا يمكن القسمة على صفر")
    return a / b

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def test_add_fails_example():
    result = add(2, 2)
    assert result == 5  # سيفشل عمدًا لنرى رسالة الخطأ

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def test_various_assertions():
    assert 5 > 3                           # مقارنة
    assert "hello" in "hello world"        # الاحتواء
    assert [1, 2, 3] == [1, 2, 3]           # تساوي القوائم
    assert isinstance(5, int)               # النوع
    assert not False                        # النفي
    assert {"a": 1} == {"a": 1}             # تساوي القواميس

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import pytest

@pytest.fixture
def sample_cart():
    """Fixture تُجهّز سلة تسوّق تجريبية لاستخدامها في عدة اختبارات."""
    return {
        "items": [
            {"name": "لابتوب", "price": 3500, "qty": 1},
            {"name": "ماوس", "price": 50, "qty": 2},
        ],
    }

def test_cart_has_two_item_types(sample_cart):
    assert len(sample_cart["items"]) == 2

def test_cart_total_quantity(sample_cart):
    total_qty = sum(item["qty"] for item in sample_cart["items"])
    assert total_qty == 3

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="function")  # الافتراضي: يُعاد إنشاؤها لكل اختبار
def fresh_data():
    return []

@pytest.fixture(scope="module")  # تُنشأ مرة واحدة فقط لكل ملف اختبار
def database_connection():
    print("🔌 فتح اتصال بقاعدة البيانات...")
    connection = "وهمي-اتصال"
    yield connection  # القيمة التي تُمرَّر للاختبارات
    print("🔌 إغلاق الاتصال...")  # كود التنظيف، يُنفَّذ بعد انتهاء كل الاختبارات

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-25.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pytest
from pathlib import Path

@pytest.fixture
def temp_file(tmp_path):
    """إنشاء ملف مؤقت للاختبار، وحذفه تلقائيًا بعد الانتهاء."""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("بيانات تجريبية", encoding="utf-8")

    yield file_path  # الاختبار يستخدم هذه القيمة

    # هذا الجزء يُنفَّذ تلقائيًا بعد انتهاء الاختبار (تنظيف)
    print(f"تنظيف: {file_path}")

def test_file_content(temp_file):
    content = temp_file.read_text(encoding="utf-8")
    assert content == "بيانات تجريبية"

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-25.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# src/bank_account.py
class InsufficientFundsError(Exception):
    """يُرفع عند محاولة سحب مبلغ أكبر من الرصيد المتاح."""
    pass

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("مبلغ الإيداع يجب أن يكون موجبًا")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFundsError("الرصيد غير كافٍ")
        self.balance -= amount

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-25.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# src/async_utils.py
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-25.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# src/weather.py
import requests

def get_temperature(city: str) -> float:
    """جلب درجة الحرارة الحالية لمدينة عبر API خارجي."""
    response = requests.get(f"https://api.weather.example.com/{city}", timeout=10)
    data = response.json()
    return data["temperature"]

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-25.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# 4
import pytest

@pytest.fixture
def sample_products():
    return [
        {"name": "أ", "price": 100},
        {"name": "ب", "price": 200},
        {"name": "ج", "price": 300},
    ]

def average_price(products: list[dict]) -> float:
    return sum(p["price"] for p in products) / len(products)

def test_average_price(sample_products):
    assert average_price(sample_products) == 200

# 5
from datetime import datetime
from unittest.mock import patch

def get_current_year() -> int:
    return datetime.now().year

def test_get_current_year_with_fixed_date():
    fixed_date = datetime(2030, 1, 1)
    with patch("__main__.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_date
        assert get_current_year() == 2030

# ────────────────────────────────────────────────────────────────────────
# [013] code/chapters/ch25/25-001-test_addition.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.2: المفاهيم الأساسية للاختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-001-test_addition.py
"""

def test_addition():
    # Arrange (التحضير)
    a = 5
    b = 3

    # Act (التنفيذ)
    result = a + b

    # Assert (التحقق)
    assert result == 8

# ────────────────────────────────────────────────────────────────────────
# [014] code/chapters/ch25/25-002-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-002-add.py
"""

# src/calculator.py
def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def divide(a: float, b: float) -> float:
    """قسمة رقمين، مع رفع استثناء عند القسمة على صفر."""
    if b == 0:
        raise ValueError("لا يمكن القسمة على صفر")
    return a / b

# ────────────────────────────────────────────────────────────────────────
# [015] code/chapters/ch25/25-003-test_add_two_positive_numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-003-test_add_two_positive_numbers.py
"""

# tests/test_calculator.py
from calculator import add, divide

def test_add_two_positive_numbers():
    result = add(2, 3)
    assert result == 5

def test_add_negative_numbers():
    result = add(-2, -3)
    assert result == -5

def test_divide_normal_case():
    result = divide(10, 2)
    assert result == 5

# ────────────────────────────────────────────────────────────────────────
# [016] code/chapters/ch25/25-004-test_add_fails_example.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-004-test_add_fails_example.py
"""

def test_add_fails_example():
    result = add(2, 2)
    assert result == 5  # سيفشل عمدًا لنرى رسالة الخطأ

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch25/25-005-test_various_assertions.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-005-test_various_assertions.py
"""

def test_various_assertions():
    assert 5 > 3                           # مقارنة
    assert "hello" in "hello world"        # الاحتواء
    assert [1, 2, 3] == [1, 2, 3]           # تساوي القوائم
    assert isinstance(5, int)               # النوع
    assert not False                        # النفي
    assert {"a": 1} == {"a": 1}             # تساوي القواميس

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch25/25-006-test_divide_by_zero_raises_error.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-006-test_divide_by_zero_raises_error.py
"""

import pytest
from calculator import divide

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="لا يمكن القسمة على صفر"):
        divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch25/25-007-test_float_addition_naive.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-007-test_float_addition_naive.py
"""

def test_float_addition_naive():
    assert 0.1 + 0.2 == 0.3  # ❌ قد يفشل! بسبب دقة الفاصلة العائمة

def test_float_addition_correct():
    assert 0.1 + 0.2 == pytest.approx(0.3)  # ✅ يتحقق بتسامح معقول من الفروق الدقيقة جدًا

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch25/25-008-sample_cart.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-008-sample_cart.py
"""

import pytest

@pytest.fixture
def sample_cart():
    """Fixture تُجهّز سلة تسوّق تجريبية لاستخدامها في عدة اختبارات."""
    return {
        "items": [
            {"name": "لابتوب", "price": 3500, "qty": 1},
            {"name": "ماوس", "price": 50, "qty": 2},
        ],
    }

def test_cart_has_two_item_types(sample_cart):
    assert len(sample_cart["items"]) == 2

def test_cart_total_quantity(sample_cart):
    total_qty = sum(item["qty"] for item in sample_cart["items"])
    assert total_qty == 3

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch25/25-009-fresh_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-009-fresh_data.py
"""

@pytest.fixture(scope="function")  # الافتراضي: يُعاد إنشاؤها لكل اختبار
def fresh_data():
    return []

@pytest.fixture(scope="module")  # تُنشأ مرة واحدة فقط لكل ملف اختبار
def database_connection():
    print("🔌 فتح اتصال بقاعدة البيانات...")
    connection = "وهمي-اتصال"
    yield connection  # القيمة التي تُمرَّر للاختبارات
    print("🔌 إغلاق الاتصال...")  # كود التنظيف، يُنفَّذ بعد انتهاء كل الاختبارات

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch25/25-010-temp_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-010-temp_file.py
"""

import pytest
from pathlib import Path

@pytest.fixture
def temp_file(tmp_path):
    """إنشاء ملف مؤقت للاختبار، وحذفه تلقائيًا بعد الانتهاء."""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("بيانات تجريبية", encoding="utf-8")

    yield file_path  # الاختبار يستخدم هذه القيمة

    # هذا الجزء يُنفَّذ تلقائيًا بعد انتهاء الاختبار (تنظيف)
    print(f"تنظيف: {file_path}")

def test_file_content(temp_file):
    content = temp_file.read_text(encoding="utf-8")
    assert content == "بيانات تجريبية"

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch25/25-011-test_add_multiple_cases.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.6: Parametrize: اختبار حالات متعددة بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-011-test_add_multiple_cases.py
"""

import pytest
from calculator import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch25/25-012-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-012-__init__.py
"""

# src/bank_account.py
class InsufficientFundsError(Exception):
    """يُرفع عند محاولة سحب مبلغ أكبر من الرصيد المتاح."""
    pass

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("مبلغ الإيداع يجب أن يكون موجبًا")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFundsError("الرصيد غير كافٍ")
        self.balance -= amount

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch25/25-013-account.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-013-account.py
"""

# tests/test_bank_account.py
import pytest
from bank_account import BankAccount, InsufficientFundsError

@pytest.fixture
def account():
    """حساب بنكي تجريبي برصيد ابتدائي 100."""
    return BankAccount(owner="أحمد", balance=100)

class TestDeposit:
    """تجميع كل اختبارات عملية الإيداع في صنف واحد."""

    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150

    def test_deposit_negative_amount_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(-10)

    def test_deposit_zero_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(0)

class TestWithdraw:
    """تجميع كل اختبارات عملية السحب في صنف واحد."""

    def test_withdraw_decreases_balance(self, account):
        account.withdraw(30)
        assert account.balance == 70

    def test_withdraw_more_than_balance_raises_error(self, account):
        with pytest.raises(InsufficientFundsError):
            account.withdraw(200)

    def test_withdraw_exact_balance_succeeds(self, account):
        account.withdraw(100)
        assert account.balance == 0

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch25/25-014-fetch_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-014-fetch_data.py
"""

# src/async_utils.py
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch25/25-015-test_fetch_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-015-test_fetch_data.py
"""

# tests/test_async_utils.py
import pytest
from async_utils import fetch_data

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch25/25-016-get_temperature.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-016-get_temperature.py
"""

# src/weather.py
import requests

def get_temperature(city: str) -> float:
    """جلب درجة الحرارة الحالية لمدينة عبر API خارجي."""
    response = requests.get(f"https://api.weather.example.com/{city}", timeout=10)
    data = response.json()
    return data["temperature"]

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch25/25-017-test_get_temperature_with_mock.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-017-test_get_temperature_with_mock.py
"""

# tests/test_weather.py
from unittest.mock import MagicMock, patch

from weather import get_temperature

def test_get_temperature_with_mock():
    # استبدال requests.get بنسخة وهمية لا تتصل فعليًا بالإنترنت
    with patch("weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25.5}
        mock_get.return_value = mock_response

        result = get_temperature("الرياض")

        assert result == 25.5
        mock_get.assert_called_once()  # التحقق من أن الدالة استُدعيت فعلاً

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch25/25-018-product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-018-product.py
"""

"""نماذج بيانات نظام إدارة المخزون."""

from dataclasses import dataclass

@dataclass
class Product:
    """يمثّل منتجًا في المخزون."""

    sku: str
    name: str
    price: float
    quantity: int

class ProductNotFoundError(Exception):
    """يُرفع عند البحث عن منتج غير موجود."""
    pass

class InsufficientStockError(Exception):
    """يُرفع عند محاولة بيع كمية أكبر من المتوفر."""
    pass

class DuplicateSKUError(Exception):
    """يُرفع عند محاولة إضافة منتج برمز SKU موجود مسبقًا."""
    pass

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch25/25-019-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-019-__init__.py
"""

"""منطق إدارة المخزون: إضافة، بيع، واستعلام عن المنتجات."""

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class InventoryManager:
    """يدير مجموعة من المنتجات داخل المخزون."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """إضافة منتج جديد إلى المخزون.

        Raises:
            DuplicateSKUError: إذا كان رمز SKU مستخدمًا مسبقًا.
        """
        if product.sku in self._products:
            raise DuplicateSKUError(f"المنتج {product.sku} موجود بالفعل")
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        """الحصول على منتج بواسطة رمزه.

        Raises:
            ProductNotFoundError: إذا لم يوجد منتج بهذا الرمز.
        """
        if sku not in self._products:
            raise ProductNotFoundError(f"المنتج {sku} غير موجود")
        return self._products[sku]

    def sell(self, sku: str, quantity: int) -> float:
        """بيع كمية من منتج، وإرجاع الإيراد الناتج.

        Raises:
            ProductNotFoundError: إذا لم يوجد المنتج.
            ValueError: إذا كانت الكمية المطلوبة غير موجبة.
            InsufficientStockError: إذا تجاوزت الكمية المطلوبة المخزون المتاح.
        """
        if quantity <= 0:
            raise ValueError("الكمية يجب أن تكون موجبة")

        product = self.get_product(sku)

        if quantity > product.quantity:
            raise InsufficientStockError(
                f"الكمية المتوفرة ({product.quantity}) أقل من المطلوبة ({quantity})"
            )

        product.quantity -= quantity
        return product.price * quantity

    def total_inventory_value(self) -> float:
        """حساب القيمة الإجمالية لكل المخزون الحالي."""
        return sum(p.price * p.quantity for p in self._products.values())

    def low_stock_products(self, threshold: int = 5) -> list[Product]:
        """إرجاع قائمة المنتجات التي كميتها أقل من أو تساوي حدًا معيّنًا."""
        return [p for p in self._products.values() if p.quantity <= threshold]

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch25/25-020-empty_manager.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-020-empty_manager.py
"""

"""Fixtures مشتركة متاحة تلقائيًا لكل ملفات الاختبار في هذا المجلد."""

import pytest

from inventory.inventory_manager import InventoryManager
from inventory.models import Product

@pytest.fixture
def empty_manager():
    """مدير مخزون فارغ تمامًا."""
    return InventoryManager()

@pytest.fixture
def manager_with_products(empty_manager):
    """مدير مخزون مُعبّأ بمنتجات تجريبية جاهزة للاختبارات."""
    empty_manager.add_product(Product(sku="LAP001", name="لابتوب", price=3500, quantity=10))
    empty_manager.add_product(Product(sku="MOU001", name="ماوس", price=50, quantity=100))
    empty_manager.add_product(Product(sku="KEY001", name="لوحة مفاتيح", price=120, quantity=3))
    return empty_manager

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch25/25-021-test_add_product_succeeds.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-021-test_add_product_succeeds.py
"""

"""اختبارات شاملة لنظام إدارة المخزون."""

import pytest

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class TestAddProduct:
    """اختبارات إضافة منتج جديد."""

    def test_add_product_succeeds(self, empty_manager):
        product = Product(sku="TEST01", name="منتج تجريبي", price=100, quantity=5)
        empty_manager.add_product(product)

        retrieved = empty_manager.get_product("TEST01")
        assert retrieved.name == "منتج تجريبي"

    def test_add_duplicate_sku_raises_error(self, manager_with_products):
        duplicate = Product(sku="LAP001", name="لابتوب آخر", price=4000, quantity=5)

        with pytest.raises(DuplicateSKUError):
            manager_with_products.add_product(duplicate)

class TestGetProduct:
    """اختبارات استرجاع منتج."""

    def test_get_existing_product(self, manager_with_products):
        product = manager_with_products.get_product("MOU001")
        assert product.name == "ماوس"
        assert product.price == 50

    def test_get_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.get_product("NOTEXIST")

class TestSellProduct:
    """اختبارات عملية البيع."""

    def test_sell_reduces_quantity(self, manager_with_products):
        manager_with_products.sell("LAP001", 3)
        remaining = manager_with_products.get_product("LAP001")
        assert remaining.quantity == 7

    def test_sell_returns_correct_revenue(self, manager_with_products):
        revenue = manager_with_products.sell("MOU001", 10)
        assert revenue == 500  # 50 * 10

    def test_sell_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.sell("NOTEXIST", 1)

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -100])
    def test_sell_non_positive_quantity_raises_error(
        self, manager_with_products, invalid_quantity
    ):
        with pytest.raises(ValueError):
            manager_with_products.sell("LAP001", invalid_quantity)

    def test_sell_more_than_available_raises_error(self, manager_with_products):
        with pytest.raises(InsufficientStockError):
            manager_with_products.sell("KEY001", 1000)

    def test_sell_exact_remaining_quantity_succeeds(self, manager_with_products):
        """حالة حدّية: بيع الكمية المتبقية بالضبط يجب أن ينجح (وليس يرفض)."""
        manager_with_products.sell("KEY001", 3)  # الكمية المتوفرة بالضبط
        remaining = manager_with_products.get_product("KEY001")
        assert remaining.quantity == 0

class TestInventoryReports:
    """اختبارات تقارير المخزون الإجمالية."""

    def test_total_inventory_value(self, manager_with_products):
        # (3500*10) + (50*100) + (120*3) = 35000 + 5000 + 360 = 40360
        assert manager_with_products.total_inventory_value() == 40360

    def test_empty_inventory_value_is_zero(self, empty_manager):
        assert empty_manager.total_inventory_value() == 0

    def test_low_stock_products_default_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products()
        skus = [p.sku for p in low_stock]
        assert "KEY001" in skus      # كمية 3، أقل من الحد الافتراضي 5
        assert "LAP001" not in skus  # كمية 10، أعلى من الحد

    def test_low_stock_products_custom_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products(threshold=15)
        skus = [p.sku for p in low_stock]
        assert "LAP001" in skus  # كمية 10، أقل من الحد المخصص 15

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch25/25-022-is_palindrome.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-022-is_palindrome.py
"""

# دوال للاختبار (ضعها في ملف منفصل src/utils.py)
def is_palindrome(text: str) -> bool:
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def calculate_average(numbers: list) -> float:
    if not numbers:
        raise ValueError("القائمة لا يمكن أن تكون فارغة")
    return sum(numbers) / len(numbers)

def is_even(n: int) -> bool:
    return n % 2 == 0

# الاختبارات
import pytest
from utils import is_palindrome, calculate_average, is_even

def test_is_palindrome_true_case():
    assert is_palindrome("مدم") is True

def test_is_palindrome_false_case():
    assert is_palindrome("بيت") is False

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("أ") is True

def test_calculate_average_empty_list_raises_error():
    with pytest.raises(ValueError):
        calculate_average([])

@pytest.mark.parametrize("n, expected", [
    (2, True), (3, False), (0, True), (-4, True), (-7, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch25/25-023-sample_products.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-023-sample_products.py
"""

# 4
import pytest

@pytest.fixture
def sample_products():
    return [
        {"name": "أ", "price": 100},
        {"name": "ب", "price": 200},
        {"name": "ج", "price": 300},
    ]

def average_price(products: list[dict]) -> float:
    return sum(p["price"] for p in products) / len(products)

def test_average_price(sample_products):
    assert average_price(sample_products) == 200

# 5
from datetime import datetime
from unittest.mock import patch

def get_current_year() -> int:
    return datetime.now().year

def test_get_current_year_with_fixed_date():
    fixed_date = datetime(2030, 1, 1)
    with patch("__main__.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_date
        assert get_current_year() == 2030

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch25/25-024-restock.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-024-restock.py
"""

# إضافة إلى inventory_manager.py
def restock(self, sku: str, quantity: int) -> None:
    """زيادة كمية منتج موجود في المخزون.

    Raises:
        ProductNotFoundError: إذا لم يوجد المنتج.
        ValueError: إذا كانت الكمية المُضافة غير موجبة.
    """
    if quantity <= 0:
        raise ValueError("كمية التزويد يجب أن تكون موجبة")

    product = self.get_product(sku)  # يرفع ProductNotFoundError تلقائيًا إن لم يوجد
    product.quantity += quantity

# الاختبارات
class TestRestock:
    def test_restock_increases_quantity(self, manager_with_products):
        manager_with_products.restock("KEY001", 10)
        product = manager_with_products.get_product("KEY001")
        assert product.quantity == 13

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -50])
    def test_restock_non_positive_raises_error(self, manager_with_products, invalid_quantity):
        with pytest.raises(ValueError):
            manager_with_products.restock("KEY001", invalid_quantity)

    def test_restock_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.restock("NOTEXIST", 10)

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch25/25-025-test_something.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.15: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-025-test_something.py
"""

# ===== الأساسيات =====
def test_something():                    # يجب أن يبدأ بـ test_
    assert actual == expected

# ===== اختبار الاستثناءات =====
import pytest
with pytest.raises(ValueError):
    risky_function()
with pytest.raises(ValueError, match="رسالة محددة"):
    risky_function()

# ===== الأعداد العشرية =====
assert 0.1 + 0.2 == pytest.approx(0.3)

# ===== Fixtures =====
@pytest.fixture
def my_data():
    return {"key": "value"}

def test_uses_fixture(my_data):
    assert my_data["key"] == "value"

@pytest.fixture
def resource_with_cleanup():
    resource = setup()
    yield resource
    teardown(resource)

# ===== Parametrize =====
@pytest.mark.parametrize("a, b, expected", [(1,1,2), (2,2,4)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# ===== Mocking =====
from unittest.mock import patch, MagicMock
with patch("module.function_name") as mock_func:
    mock_func.return_value = "قيمة وهمية"

# ===== أوامر التشغيل =====
# uv run pytest                              تشغيل كل الاختبارات
# uv run pytest -v                           تفصيلي
# uv run pytest tests/test_x.py::test_y      اختبار محدد
# uv run pytest --cov=src                    مع تقرير التغطية
# uv run pytest -k "keyword"                 فلترة بالاسم

# ────────────────────────────────────────────────────────────────────────
# [038] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# tests/test_calculator.py
from calculator import add, divide

def test_add_two_positive_numbers():
    result = add(2, 3)
    assert result == 5

def test_add_negative_numbers():
    result = add(-2, -3)
    assert result == -5

def test_divide_normal_case():
    result = divide(10, 2)
    assert result == 5

# ────────────────────────────────────────────────────────────────────────
# [039] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
FAILED tests/test_calculator.py::test_add_fails_example - assert 4 == 5
    def test_add_fails_example():
        result = add(2, 2)
>       assert result == 5
E       assert 4 == 5

# ────────────────────────────────────────────────────────────────────────
# [040] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
import pytest
from calculator import divide

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="لا يمكن القسمة على صفر"):
        divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [041] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
def test_float_addition_naive():
    assert 0.1 + 0.2 == 0.3  # ❌ قد يفشل! بسبب دقة الفاصلة العائمة

def test_float_addition_correct():
    assert 0.1 + 0.2 == pytest.approx(0.3)  # ✅ يتحقق بتسامح معقول من الفروق الدقيقة جدًا

# ────────────────────────────────────────────────────────────────────────
# [042] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
import pytest
from calculator import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected

# ────────────────────────────────────────────────────────────────────────
# [043] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# tests/test_bank_account.py
import pytest
from bank_account import BankAccount, InsufficientFundsError

@pytest.fixture
def account():
    """حساب بنكي تجريبي برصيد ابتدائي 100."""
    return BankAccount(owner="أحمد", balance=100)

class TestDeposit:
    """تجميع كل اختبارات عملية الإيداع في صنف واحد."""

    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150

    def test_deposit_negative_amount_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(-10)

    def test_deposit_zero_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(0)

class TestWithdraw:
    """تجميع كل اختبارات عملية السحب في صنف واحد."""

    def test_withdraw_decreases_balance(self, account):
        account.withdraw(30)
        assert account.balance == 70

    def test_withdraw_more_than_balance_raises_error(self, account):
        with pytest.raises(InsufficientFundsError):
            account.withdraw(200)

    def test_withdraw_exact_balance_succeeds(self, account):
        account.withdraw(100)
        assert account.balance == 0

# ────────────────────────────────────────────────────────────────────────
# [044] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# tests/test_async_utils.py
import pytest
from async_utils import fetch_data

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [045] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# tests/test_weather.py
from unittest.mock import MagicMock, patch

from weather import get_temperature

def test_get_temperature_with_mock():
    # استبدال requests.get بنسخة وهمية لا تتصل فعليًا بالإنترنت
    with patch("weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25.5}
        mock_get.return_value = mock_response

        result = get_temperature("الرياض")

        assert result == 25.5
        mock_get.assert_called_once()  # التحقق من أن الدالة استُدعيت فعلاً

# ────────────────────────────────────────────────────────────────────────
# [046] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""نماذج بيانات نظام إدارة المخزون."""

from dataclasses import dataclass

@dataclass
class Product:
    """يمثّل منتجًا في المخزون."""

    sku: str
    name: str
    price: float
    quantity: int

class ProductNotFoundError(Exception):
    """يُرفع عند البحث عن منتج غير موجود."""
    pass

class InsufficientStockError(Exception):
    """يُرفع عند محاولة بيع كمية أكبر من المتوفر."""
    pass

class DuplicateSKUError(Exception):
    """يُرفع عند محاولة إضافة منتج برمز SKU موجود مسبقًا."""
    pass

# ────────────────────────────────────────────────────────────────────────
# [047] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""منطق إدارة المخزون: إضافة، بيع، واستعلام عن المنتجات."""

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class InventoryManager:
    """يدير مجموعة من المنتجات داخل المخزون."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """إضافة منتج جديد إلى المخزون.

        Raises:
            DuplicateSKUError: إذا كان رمز SKU مستخدمًا مسبقًا.
        """
        if product.sku in self._products:
            raise DuplicateSKUError(f"المنتج {product.sku} موجود بالفعل")
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        """الحصول على منتج بواسطة رمزه.

        Raises:
            ProductNotFoundError: إذا لم يوجد منتج بهذا الرمز.
        """
        if sku not in self._products:
            raise ProductNotFoundError(f"المنتج {sku} غير موجود")
        return self._products[sku]

    def sell(self, sku: str, quantity: int) -> float:
        """بيع كمية من منتج، وإرجاع الإيراد الناتج.

        Raises:
            ProductNotFoundError: إذا لم يوجد المنتج.
            ValueError: إذا كانت الكمية المطلوبة غير موجبة.
            InsufficientStockError: إذا تجاوزت الكمية المطلوبة المخزون المتاح.
        """
        if quantity <= 0:
            raise ValueError("الكمية يجب أن تكون موجبة")

        product = self.get_product(sku)

        if quantity > product.quantity:
            raise InsufficientStockError(
                f"الكمية المتوفرة ({product.quantity}) أقل من المطلوبة ({quantity})"
            )

        product.quantity -= quantity
        return product.price * quantity

    def total_inventory_value(self) -> float:
        """حساب القيمة الإجمالية لكل المخزون الحالي."""
        return sum(p.price * p.quantity for p in self._products.values())

    def low_stock_products(self, threshold: int = 5) -> list[Product]:
        """إرجاع قائمة المنتجات التي كميتها أقل من أو تساوي حدًا معيّنًا."""
        return [p for p in self._products.values() if p.quantity <= threshold]

# ────────────────────────────────────────────────────────────────────────
# [048] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""Fixtures مشتركة متاحة تلقائيًا لكل ملفات الاختبار في هذا المجلد."""

import pytest

from inventory.inventory_manager import InventoryManager
from inventory.models import Product

@pytest.fixture
def empty_manager():
    """مدير مخزون فارغ تمامًا."""
    return InventoryManager()

@pytest.fixture
def manager_with_products(empty_manager):
    """مدير مخزون مُعبّأ بمنتجات تجريبية جاهزة للاختبارات."""
    empty_manager.add_product(Product(sku="LAP001", name="لابتوب", price=3500, quantity=10))
    empty_manager.add_product(Product(sku="MOU001", name="ماوس", price=50, quantity=100))
    empty_manager.add_product(Product(sku="KEY001", name="لوحة مفاتيح", price=120, quantity=3))
    return empty_manager

# ────────────────────────────────────────────────────────────────────────
# [049] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""اختبارات شاملة لنظام إدارة المخزون."""

import pytest

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class TestAddProduct:
    """اختبارات إضافة منتج جديد."""

    def test_add_product_succeeds(self, empty_manager):
        product = Product(sku="TEST01", name="منتج تجريبي", price=100, quantity=5)
        empty_manager.add_product(product)

        retrieved = empty_manager.get_product("TEST01")
        assert retrieved.name == "منتج تجريبي"

    def test_add_duplicate_sku_raises_error(self, manager_with_products):
        duplicate = Product(sku="LAP001", name="لابتوب آخر", price=4000, quantity=5)

        with pytest.raises(DuplicateSKUError):
            manager_with_products.add_product(duplicate)

class TestGetProduct:
    """اختبارات استرجاع منتج."""

    def test_get_existing_product(self, manager_with_products):
        product = manager_with_products.get_product("MOU001")
        assert product.name == "ماوس"
        assert product.price == 50

    def test_get_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.get_product("NOTEXIST")

class TestSellProduct:
    """اختبارات عملية البيع."""

    def test_sell_reduces_quantity(self, manager_with_products):
        manager_with_products.sell("LAP001", 3)
        remaining = manager_with_products.get_product("LAP001")
        assert remaining.quantity == 7

    def test_sell_returns_correct_revenue(self, manager_with_products):
        revenue = manager_with_products.sell("MOU001", 10)
        assert revenue == 500  # 50 * 10

    def test_sell_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.sell("NOTEXIST", 1)

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -100])
    def test_sell_non_positive_quantity_raises_error(
        self, manager_with_products, invalid_quantity
    ):
        with pytest.raises(ValueError):
            manager_with_products.sell("LAP001", invalid_quantity)

    def test_sell_more_than_available_raises_error(self, manager_with_products):
        with pytest.raises(InsufficientStockError):
            manager_with_products.sell("KEY001", 1000)

    def test_sell_exact_remaining_quantity_succeeds(self, manager_with_products):
        """حالة حدّية: بيع الكمية المتبقية بالضبط يجب أن ينجح (وليس يرفض)."""
        manager_with_products.sell("KEY001", 3)  # الكمية المتوفرة بالضبط
        remaining = manager_with_products.get_product("KEY001")
        assert remaining.quantity == 0

class TestInventoryReports:
    """اختبارات تقارير المخزون الإجمالية."""

    def test_total_inventory_value(self, manager_with_products):
        # (3500*10) + (50*100) + (120*3) = 35000 + 5000 + 360 = 40360
        assert manager_with_products.total_inventory_value() == 40360

    def test_empty_inventory_value_is_zero(self, empty_manager):
        assert empty_manager.total_inventory_value() == 0

    def test_low_stock_products_default_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products()
        skus = [p.sku for p in low_stock]
        assert "KEY001" in skus      # كمية 3، أقل من الحد الافتراضي 5
        assert "LAP001" not in skus  # كمية 10، أعلى من الحد

    def test_low_stock_products_custom_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products(threshold=15)
        skus = [p.sku for p in low_stock]
        assert "LAP001" in skus  # كمية 10، أقل من الحد المخصص 15

# ────────────────────────────────────────────────────────────────────────
# [050] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# دوال للاختبار (ضعها في ملف منفصل src/utils.py)
def is_palindrome(text: str) -> bool:
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def calculate_average(numbers: list) -> float:
    if not numbers:
        raise ValueError("القائمة لا يمكن أن تكون فارغة")
    return sum(numbers) / len(numbers)

def is_even(n: int) -> bool:
    return n % 2 == 0

# الاختبارات
import pytest
from utils import is_palindrome, calculate_average, is_even

def test_is_palindrome_true_case():
    assert is_palindrome("مدم") is True

def test_is_palindrome_false_case():
    assert is_palindrome("بيت") is False

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("أ") is True

def test_calculate_average_empty_list_raises_error():
    with pytest.raises(ValueError):
        calculate_average([])

@pytest.mark.parametrize("n, expected", [
    (2, True), (3, False), (0, True), (-4, True), (-7, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected

# ────────────────────────────────────────────────────────────────────────
# [051] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# إضافة إلى inventory_manager.py
def restock(self, sku: str, quantity: int) -> None:
    """زيادة كمية منتج موجود في المخزون.

    Raises:
        ProductNotFoundError: إذا لم يوجد المنتج.
        ValueError: إذا كانت الكمية المُضافة غير موجبة.
    """
    if quantity <= 0:
        raise ValueError("كمية التزويد يجب أن تكون موجبة")

    product = self.get_product(sku)  # يرفع ProductNotFoundError تلقائيًا إن لم يوجد
    product.quantity += quantity

# الاختبارات
class TestRestock:
    def test_restock_increases_quantity(self, manager_with_products):
        manager_with_products.restock("KEY001", 10)
        product = manager_with_products.get_product("KEY001")
        assert product.quantity == 13

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -50])
    def test_restock_non_positive_raises_error(self, manager_with_products, invalid_quantity):
        with pytest.raises(ValueError):
            manager_with_products.restock("KEY001", invalid_quantity)

    def test_restock_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.restock("NOTEXIST", 10)

# ────────────────────────────────────────────────────────────────────────
# [052] dist/markdown/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ===== الأساسيات =====
def test_something():                    # يجب أن يبدأ بـ test_
    assert actual == expected

# ===== اختبار الاستثناءات =====
import pytest
with pytest.raises(ValueError):
    risky_function()
with pytest.raises(ValueError, match="رسالة محددة"):
    risky_function()

# ===== الأعداد العشرية =====
assert 0.1 + 0.2 == pytest.approx(0.3)

# ===== Fixtures =====
@pytest.fixture
def my_data():
    return {"key": "value"}

def test_uses_fixture(my_data):
    assert my_data["key"] == "value"

@pytest.fixture
def resource_with_cleanup():
    resource = setup()
    yield resource
    teardown(resource)

# ===== Parametrize =====
@pytest.mark.parametrize("a, b, expected", [(1,1,2), (2,2,4)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# ===== Mocking =====
from unittest.mock import patch, MagicMock
with patch("module.function_name") as mock_func:
    mock_func.return_value = "قيمة وهمية"

# ===== أوامر التشغيل =====
# uv run pytest                              تشغيل كل الاختبارات
# uv run pytest -v                           تفصيلي
# uv run pytest tests/test_x.py::test_y      اختبار محدد
# uv run pytest --cov=src                    مع تقرير التغطية
# uv run pytest -k "keyword"                 فلترة بالاسم

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["pytest"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q pytest

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def test_addition():
    # Arrange (التحضير)
    a = 5
    b = 3

    # Act (التنفيذ)
    result = a + b

    # Assert (التحقق)
    assert result == 8

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# src/calculator.py
def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def divide(a: float, b: float) -> float:
    """قسمة رقمين، مع رفع استثناء عند القسمة على صفر."""
    if b == 0:
        raise ValueError("لا يمكن القسمة على صفر")
    return a / b

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def test_add_fails_example():
    result = add(2, 2)
    assert result == 5  # سيفشل عمدًا لنرى رسالة الخطأ

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def test_various_assertions():
    assert 5 > 3                           # مقارنة
    assert "hello" in "hello world"        # الاحتواء
    assert [1, 2, 3] == [1, 2, 3]           # تساوي القوائم
    assert isinstance(5, int)               # النوع
    assert not False                        # النفي
    assert {"a": 1} == {"a": 1}             # تساوي القواميس

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import pytest

@pytest.fixture
def sample_cart():
    """Fixture تُجهّز سلة تسوّق تجريبية لاستخدامها في عدة اختبارات."""
    return {
        "items": [
            {"name": "لابتوب", "price": 3500, "qty": 1},
            {"name": "ماوس", "price": 50, "qty": 2},
        ],
    }

def test_cart_has_two_item_types(sample_cart):
    assert len(sample_cart["items"]) == 2

def test_cart_total_quantity(sample_cart):
    total_qty = sum(item["qty"] for item in sample_cart["items"])
    assert total_qty == 3

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="function")  # الافتراضي: يُعاد إنشاؤها لكل اختبار
def fresh_data():
    return []

@pytest.fixture(scope="module")  # تُنشأ مرة واحدة فقط لكل ملف اختبار
def database_connection():
    print("🔌 فتح اتصال بقاعدة البيانات...")
    connection = "وهمي-اتصال"
    yield connection  # القيمة التي تُمرَّر للاختبارات
    print("🔌 إغلاق الاتصال...")  # كود التنظيف، يُنفَّذ بعد انتهاء كل الاختبارات

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pytest
from pathlib import Path

@pytest.fixture
def temp_file(tmp_path):
    """إنشاء ملف مؤقت للاختبار، وحذفه تلقائيًا بعد الانتهاء."""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("بيانات تجريبية", encoding="utf-8")

    yield file_path  # الاختبار يستخدم هذه القيمة

    # هذا الجزء يُنفَّذ تلقائيًا بعد انتهاء الاختبار (تنظيف)
    print(f"تنظيف: {file_path}")

def test_file_content(temp_file):
    content = temp_file.read_text(encoding="utf-8")
    assert content == "بيانات تجريبية"

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# src/bank_account.py
class InsufficientFundsError(Exception):
    """يُرفع عند محاولة سحب مبلغ أكبر من الرصيد المتاح."""
    pass

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("مبلغ الإيداع يجب أن يكون موجبًا")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFundsError("الرصيد غير كافٍ")
        self.balance -= amount

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# src/async_utils.py
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# src/weather.py
import requests

def get_temperature(city: str) -> float:
    """جلب درجة الحرارة الحالية لمدينة عبر API خارجي."""
    response = requests.get(f"https://api.weather.example.com/{city}", timeout=10)
    data = response.json()
    return data["temperature"]

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# 4
import pytest

@pytest.fixture
def sample_products():
    return [
        {"name": "أ", "price": 100},
        {"name": "ب", "price": 200},
        {"name": "ج", "price": 300},
    ]

def average_price(products: list[dict]) -> float:
    return sum(p["price"] for p in products) / len(products)

def test_average_price(sample_products):
    assert average_price(sample_products) == 200

# 5
from datetime import datetime
from unittest.mock import patch

def get_current_year() -> int:
    return datetime.now().year

def test_get_current_year_with_fixed_date():
    fixed_date = datetime(2030, 1, 1)
    with patch("__main__.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_date
        assert get_current_year() == 2030

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.2: المفاهيم الأساسية للاختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-001-test_addition.py
"""

def test_addition():
    # Arrange (التحضير)
    a = 5
    b = 3

    # Act (التنفيذ)
    result = a + b

    # Assert (التحقق)
    assert result == 8

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-002-add.py
"""

# src/calculator.py
def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def divide(a: float, b: float) -> float:
    """قسمة رقمين، مع رفع استثناء عند القسمة على صفر."""
    if b == 0:
        raise ValueError("لا يمكن القسمة على صفر")
    return a / b

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.3: البدء مع pytest

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-003-test_add_two_positive_numbers.py
"""

# tests/test_calculator.py
from calculator import add, divide

def test_add_two_positive_numbers():
    result = add(2, 3)
    assert result == 5

def test_add_negative_numbers():
    result = add(-2, -3)
    assert result == -5

def test_divide_normal_case():
    result = divide(10, 2)
    assert result == 5

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-004-test_add_fails_example.py
"""

def test_add_fails_example():
    result = add(2, 2)
    assert result == 5  # سيفشل عمدًا لنرى رسالة الخطأ

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-005-test_various_assertions.py
"""

def test_various_assertions():
    assert 5 > 3                           # مقارنة
    assert "hello" in "hello world"        # الاحتواء
    assert [1, 2, 3] == [1, 2, 3]           # تساوي القوائم
    assert isinstance(5, int)               # النوع
    assert not False                        # النفي
    assert {"a": 1} == {"a": 1}             # تساوي القواميس

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-006-test_divide_by_zero_raises_error.py
"""

import pytest
from calculator import divide

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="لا يمكن القسمة على صفر"):
        divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.4: `assert`: قلب كل اختبار

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-007-test_float_addition_naive.py
"""

def test_float_addition_naive():
    assert 0.1 + 0.2 == 0.3  # ❌ قد يفشل! بسبب دقة الفاصلة العائمة

def test_float_addition_correct():
    assert 0.1 + 0.2 == pytest.approx(0.3)  # ✅ يتحقق بتسامح معقول من الفروق الدقيقة جدًا

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-008-sample_cart.py
"""

import pytest

@pytest.fixture
def sample_cart():
    """Fixture تُجهّز سلة تسوّق تجريبية لاستخدامها في عدة اختبارات."""
    return {
        "items": [
            {"name": "لابتوب", "price": 3500, "qty": 1},
            {"name": "ماوس", "price": 50, "qty": 2},
        ],
    }

def test_cart_has_two_item_types(sample_cart):
    assert len(sample_cart["items"]) == 2

def test_cart_total_quantity(sample_cart):
    total_qty = sum(item["qty"] for item in sample_cart["items"])
    assert total_qty == 3

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-009-fresh_data.py
"""

@pytest.fixture(scope="function")  # الافتراضي: يُعاد إنشاؤها لكل اختبار
def fresh_data():
    return []

@pytest.fixture(scope="module")  # تُنشأ مرة واحدة فقط لكل ملف اختبار
def database_connection():
    print("🔌 فتح اتصال بقاعدة البيانات...")
    connection = "وهمي-اتصال"
    yield connection  # القيمة التي تُمرَّر للاختبارات
    print("🔌 إغلاق الاتصال...")  # كود التنظيف، يُنفَّذ بعد انتهاء كل الاختبارات

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.5: Fixtures: تجهيز بيانات الاختبار بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-010-temp_file.py
"""

import pytest
from pathlib import Path

@pytest.fixture
def temp_file(tmp_path):
    """إنشاء ملف مؤقت للاختبار، وحذفه تلقائيًا بعد الانتهاء."""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("بيانات تجريبية", encoding="utf-8")

    yield file_path  # الاختبار يستخدم هذه القيمة

    # هذا الجزء يُنفَّذ تلقائيًا بعد انتهاء الاختبار (تنظيف)
    print(f"تنظيف: {file_path}")

def test_file_content(temp_file):
    content = temp_file.read_text(encoding="utf-8")
    assert content == "بيانات تجريبية"

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.6: Parametrize: اختبار حالات متعددة بكفاءة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-011-test_add_multiple_cases.py
"""

import pytest
from calculator import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-012-__init__.py
"""

# src/bank_account.py
class InsufficientFundsError(Exception):
    """يُرفع عند محاولة سحب مبلغ أكبر من الرصيد المتاح."""
    pass

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("مبلغ الإيداع يجب أن يكون موجبًا")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFundsError("الرصيد غير كافٍ")
        self.balance -= amount

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-013-account.py
"""

# tests/test_bank_account.py
import pytest
from bank_account import BankAccount, InsufficientFundsError

@pytest.fixture
def account():
    """حساب بنكي تجريبي برصيد ابتدائي 100."""
    return BankAccount(owner="أحمد", balance=100)

class TestDeposit:
    """تجميع كل اختبارات عملية الإيداع في صنف واحد."""

    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150

    def test_deposit_negative_amount_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(-10)

    def test_deposit_zero_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(0)

class TestWithdraw:
    """تجميع كل اختبارات عملية السحب في صنف واحد."""

    def test_withdraw_decreases_balance(self, account):
        account.withdraw(30)
        assert account.balance == 70

    def test_withdraw_more_than_balance_raises_error(self, account):
        with pytest.raises(InsufficientFundsError):
            account.withdraw(200)

    def test_withdraw_exact_balance_succeeds(self, account):
        account.withdraw(100)
        assert account.balance == 0

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-014-fetch_data.py
"""

# src/async_utils.py
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-015-test_fetch_data.py
"""

# tests/test_async_utils.py
import pytest
from async_utils import fetch_data

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-016-get_temperature.py
"""

# src/weather.py
import requests

def get_temperature(city: str) -> float:
    """جلب درجة الحرارة الحالية لمدينة عبر API خارجي."""
    response = requests.get(f"https://api.weather.example.com/{city}", timeout=10)
    data = response.json()
    return data["temperature"]

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.10: Mocking: محاكاة الأجزاء الخارجية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-017-test_get_temperature_with_mock.py
"""

# tests/test_weather.py
from unittest.mock import MagicMock, patch

from weather import get_temperature

def test_get_temperature_with_mock():
    # استبدال requests.get بنسخة وهمية لا تتصل فعليًا بالإنترنت
    with patch("weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25.5}
        mock_get.return_value = mock_response

        result = get_temperature("الرياض")

        assert result == 25.5
        mock_get.assert_called_once()  # التحقق من أن الدالة استُدعيت فعلاً

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-018-product.py
"""

"""نماذج بيانات نظام إدارة المخزون."""

from dataclasses import dataclass

@dataclass
class Product:
    """يمثّل منتجًا في المخزون."""

    sku: str
    name: str
    price: float
    quantity: int

class ProductNotFoundError(Exception):
    """يُرفع عند البحث عن منتج غير موجود."""
    pass

class InsufficientStockError(Exception):
    """يُرفع عند محاولة بيع كمية أكبر من المتوفر."""
    pass

class DuplicateSKUError(Exception):
    """يُرفع عند محاولة إضافة منتج برمز SKU موجود مسبقًا."""
    pass

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-019-__init__.py
"""

"""منطق إدارة المخزون: إضافة، بيع، واستعلام عن المنتجات."""

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class InventoryManager:
    """يدير مجموعة من المنتجات داخل المخزون."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """إضافة منتج جديد إلى المخزون.

        Raises:
            DuplicateSKUError: إذا كان رمز SKU مستخدمًا مسبقًا.
        """
        if product.sku in self._products:
            raise DuplicateSKUError(f"المنتج {product.sku} موجود بالفعل")
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        """الحصول على منتج بواسطة رمزه.

        Raises:
            ProductNotFoundError: إذا لم يوجد منتج بهذا الرمز.
        """
        if sku not in self._products:
            raise ProductNotFoundError(f"المنتج {sku} غير موجود")
        return self._products[sku]

    def sell(self, sku: str, quantity: int) -> float:
        """بيع كمية من منتج، وإرجاع الإيراد الناتج.

        Raises:
            ProductNotFoundError: إذا لم يوجد المنتج.
            ValueError: إذا كانت الكمية المطلوبة غير موجبة.
            InsufficientStockError: إذا تجاوزت الكمية المطلوبة المخزون المتاح.
        """
        if quantity <= 0:
            raise ValueError("الكمية يجب أن تكون موجبة")

        product = self.get_product(sku)

        if quantity > product.quantity:
            raise InsufficientStockError(
                f"الكمية المتوفرة ({product.quantity}) أقل من المطلوبة ({quantity})"
            )

        product.quantity -= quantity
        return product.price * quantity

    def total_inventory_value(self) -> float:
        """حساب القيمة الإجمالية لكل المخزون الحالي."""
        return sum(p.price * p.quantity for p in self._products.values())

    def low_stock_products(self, threshold: int = 5) -> list[Product]:
        """إرجاع قائمة المنتجات التي كميتها أقل من أو تساوي حدًا معيّنًا."""
        return [p for p in self._products.values() if p.quantity <= threshold]

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-020-empty_manager.py
"""

"""Fixtures مشتركة متاحة تلقائيًا لكل ملفات الاختبار في هذا المجلد."""

import pytest

from inventory.inventory_manager import InventoryManager
from inventory.models import Product

@pytest.fixture
def empty_manager():
    """مدير مخزون فارغ تمامًا."""
    return InventoryManager()

@pytest.fixture
def manager_with_products(empty_manager):
    """مدير مخزون مُعبّأ بمنتجات تجريبية جاهزة للاختبارات."""
    empty_manager.add_product(Product(sku="LAP001", name="لابتوب", price=3500, quantity=10))
    empty_manager.add_product(Product(sku="MOU001", name="ماوس", price=50, quantity=100))
    empty_manager.add_product(Product(sku="KEY001", name="لوحة مفاتيح", price=120, quantity=3))
    return empty_manager

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.11: مشروع تطبيقي متكامل: اختبار نظام إدارة مخزون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-021-test_add_product_succeeds.py
"""

"""اختبارات شاملة لنظام إدارة المخزون."""

import pytest

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class TestAddProduct:
    """اختبارات إضافة منتج جديد."""

    def test_add_product_succeeds(self, empty_manager):
        product = Product(sku="TEST01", name="منتج تجريبي", price=100, quantity=5)
        empty_manager.add_product(product)

        retrieved = empty_manager.get_product("TEST01")
        assert retrieved.name == "منتج تجريبي"

    def test_add_duplicate_sku_raises_error(self, manager_with_products):
        duplicate = Product(sku="LAP001", name="لابتوب آخر", price=4000, quantity=5)

        with pytest.raises(DuplicateSKUError):
            manager_with_products.add_product(duplicate)

class TestGetProduct:
    """اختبارات استرجاع منتج."""

    def test_get_existing_product(self, manager_with_products):
        product = manager_with_products.get_product("MOU001")
        assert product.name == "ماوس"
        assert product.price == 50

    def test_get_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.get_product("NOTEXIST")

class TestSellProduct:
    """اختبارات عملية البيع."""

    def test_sell_reduces_quantity(self, manager_with_products):
        manager_with_products.sell("LAP001", 3)
        remaining = manager_with_products.get_product("LAP001")
        assert remaining.quantity == 7

    def test_sell_returns_correct_revenue(self, manager_with_products):
        revenue = manager_with_products.sell("MOU001", 10)
        assert revenue == 500  # 50 * 10

    def test_sell_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.sell("NOTEXIST", 1)

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -100])
    def test_sell_non_positive_quantity_raises_error(
        self, manager_with_products, invalid_quantity
    ):
        with pytest.raises(ValueError):
            manager_with_products.sell("LAP001", invalid_quantity)

    def test_sell_more_than_available_raises_error(self, manager_with_products):
        with pytest.raises(InsufficientStockError):
            manager_with_products.sell("KEY001", 1000)

    def test_sell_exact_remaining_quantity_succeeds(self, manager_with_products):
        """حالة حدّية: بيع الكمية المتبقية بالضبط يجب أن ينجح (وليس يرفض)."""
        manager_with_products.sell("KEY001", 3)  # الكمية المتوفرة بالضبط
        remaining = manager_with_products.get_product("KEY001")
        assert remaining.quantity == 0

class TestInventoryReports:
    """اختبارات تقارير المخزون الإجمالية."""

    def test_total_inventory_value(self, manager_with_products):
        # (3500*10) + (50*100) + (120*3) = 35000 + 5000 + 360 = 40360
        assert manager_with_products.total_inventory_value() == 40360

    def test_empty_inventory_value_is_zero(self, empty_manager):
        assert empty_manager.total_inventory_value() == 0

    def test_low_stock_products_default_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products()
        skus = [p.sku for p in low_stock]
        assert "KEY001" in skus      # كمية 3، أقل من الحد الافتراضي 5
        assert "LAP001" not in skus  # كمية 10، أعلى من الحد

    def test_low_stock_products_custom_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products(threshold=15)
        skus = [p.sku for p in low_stock]
        assert "LAP001" in skus  # كمية 10، أقل من الحد المخصص 15

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-022-is_palindrome.py
"""

# دوال للاختبار (ضعها في ملف منفصل src/utils.py)
def is_palindrome(text: str) -> bool:
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def calculate_average(numbers: list) -> float:
    if not numbers:
        raise ValueError("القائمة لا يمكن أن تكون فارغة")
    return sum(numbers) / len(numbers)

def is_even(n: int) -> bool:
    return n % 2 == 0

# الاختبارات
import pytest
from utils import is_palindrome, calculate_average, is_even

def test_is_palindrome_true_case():
    assert is_palindrome("مدم") is True

def test_is_palindrome_false_case():
    assert is_palindrome("بيت") is False

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("أ") is True

def test_calculate_average_empty_list_raises_error():
    with pytest.raises(ValueError):
        calculate_average([])

@pytest.mark.parametrize("n, expected", [
    (2, True), (3, False), (0, True), (-4, True), (-7, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-023-sample_products.py
"""

# 4
import pytest

@pytest.fixture
def sample_products():
    return [
        {"name": "أ", "price": 100},
        {"name": "ب", "price": 200},
        {"name": "ج", "price": 300},
    ]

def average_price(products: list[dict]) -> float:
    return sum(p["price"] for p in products) / len(products)

def test_average_price(sample_products):
    assert average_price(sample_products) == 200

# 5
from datetime import datetime
from unittest.mock import patch

def get_current_year() -> int:
    return datetime.now().year

def test_get_current_year_with_fixed_date():
    fixed_date = datetime(2030, 1, 1)
    with patch("__main__.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_date
        assert get_current_year() == 2030

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.13: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-024-restock.py
"""

# إضافة إلى inventory_manager.py
def restock(self, sku: str, quantity: int) -> None:
    """زيادة كمية منتج موجود في المخزون.

    Raises:
        ProductNotFoundError: إذا لم يوجد المنتج.
        ValueError: إذا كانت الكمية المُضافة غير موجبة.
    """
    if quantity <= 0:
        raise ValueError("كمية التزويد يجب أن تكون موجبة")

    product = self.get_product(sku)  # يرفع ProductNotFoundError تلقائيًا إن لم يوجد
    product.quantity += quantity

# الاختبارات
class TestRestock:
    def test_restock_increases_quantity(self, manager_with_products):
        manager_with_products.restock("KEY001", 10)
        product = manager_with_products.get_product("KEY001")
        assert product.quantity == 13

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -50])
    def test_restock_non_positive_raises_error(self, manager_with_products, invalid_quantity):
        with pytest.raises(ValueError):
            manager_with_products.restock("KEY001", invalid_quantity)

    def test_restock_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.restock("NOTEXIST", 10)

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.15: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-025-test_something.py
"""

# ===== الأساسيات =====
def test_something():                    # يجب أن يبدأ بـ test_
    assert actual == expected

# ===== اختبار الاستثناءات =====
import pytest
with pytest.raises(ValueError):
    risky_function()
with pytest.raises(ValueError, match="رسالة محددة"):
    risky_function()

# ===== الأعداد العشرية =====
assert 0.1 + 0.2 == pytest.approx(0.3)

# ===== Fixtures =====
@pytest.fixture
def my_data():
    return {"key": "value"}

def test_uses_fixture(my_data):
    assert my_data["key"] == "value"

@pytest.fixture
def resource_with_cleanup():
    resource = setup()
    yield resource
    teardown(resource)

# ===== Parametrize =====
@pytest.mark.parametrize("a, b, expected", [(1,1,2), (2,2,4)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# ===== Mocking =====
from unittest.mock import patch, MagicMock
with patch("module.function_name") as mock_func:
    mock_func.return_value = "قيمة وهمية"

# ===== أوامر التشغيل =====
# uv run pytest                              تشغيل كل الاختبارات
# uv run pytest -v                           تفصيلي
# uv run pytest tests/test_x.py::test_y      اختبار محدد
# uv run pytest --cov=src                    مع تقرير التغطية
# uv run pytest -k "keyword"                 فلترة بالاسم

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
# tests/test_calculator.py
from calculator import add, divide

def test_add_two_positive_numbers():
    result = add(2, 3)
    assert result == 5

def test_add_negative_numbers():
    result = add(-2, -3)
    assert result == -5

def test_divide_normal_case():
    result = divide(10, 2)
    assert result == 5

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
FAILED tests/test_calculator.py::test_add_fails_example - assert 4 == 5
    def test_add_fails_example():
        result = add(2, 2)
>       assert result == 5
E       assert 4 == 5

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
import pytest
from calculator import divide

def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="لا يمكن القسمة على صفر"):
        divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
def test_float_addition_naive():
    assert 0.1 + 0.2 == 0.3  # ❌ قد يفشل! بسبب دقة الفاصلة العائمة

def test_float_addition_correct():
    assert 0.1 + 0.2 == pytest.approx(0.3)  # ✅ يتحقق بتسامح معقول من الفروق الدقيقة جدًا

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
import pytest
from calculator import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
# tests/test_bank_account.py
import pytest
from bank_account import BankAccount, InsufficientFundsError

@pytest.fixture
def account():
    """حساب بنكي تجريبي برصيد ابتدائي 100."""
    return BankAccount(owner="أحمد", balance=100)

class TestDeposit:
    """تجميع كل اختبارات عملية الإيداع في صنف واحد."""

    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150

    def test_deposit_negative_amount_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(-10)

    def test_deposit_zero_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(0)

class TestWithdraw:
    """تجميع كل اختبارات عملية السحب في صنف واحد."""

    def test_withdraw_decreases_balance(self, account):
        account.withdraw(30)
        assert account.balance == 70

    def test_withdraw_more_than_balance_raises_error(self, account):
        with pytest.raises(InsufficientFundsError):
            account.withdraw(200)

    def test_withdraw_exact_balance_succeeds(self, account):
        account.withdraw(100)
        assert account.balance == 0

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
# tests/test_async_utils.py
import pytest
from async_utils import fetch_data

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "بيانات"

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
# tests/test_weather.py
from unittest.mock import MagicMock, patch

from weather import get_temperature

def test_get_temperature_with_mock():
    # استبدال requests.get بنسخة وهمية لا تتصل فعليًا بالإنترنت
    with patch("weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25.5}
        mock_get.return_value = mock_response

        result = get_temperature("الرياض")

        assert result == 25.5
        mock_get.assert_called_once()  # التحقق من أن الدالة استُدعيت فعلاً

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
"""نماذج بيانات نظام إدارة المخزون."""

from dataclasses import dataclass

@dataclass
class Product:
    """يمثّل منتجًا في المخزون."""

    sku: str
    name: str
    price: float
    quantity: int

class ProductNotFoundError(Exception):
    """يُرفع عند البحث عن منتج غير موجود."""
    pass

class InsufficientStockError(Exception):
    """يُرفع عند محاولة بيع كمية أكبر من المتوفر."""
    pass

class DuplicateSKUError(Exception):
    """يُرفع عند محاولة إضافة منتج برمز SKU موجود مسبقًا."""
    pass

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
"""منطق إدارة المخزون: إضافة، بيع، واستعلام عن المنتجات."""

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class InventoryManager:
    """يدير مجموعة من المنتجات داخل المخزون."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """إضافة منتج جديد إلى المخزون.

        Raises:
            DuplicateSKUError: إذا كان رمز SKU مستخدمًا مسبقًا.
        """
        if product.sku in self._products:
            raise DuplicateSKUError(f"المنتج {product.sku} موجود بالفعل")
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        """الحصول على منتج بواسطة رمزه.

        Raises:
            ProductNotFoundError: إذا لم يوجد منتج بهذا الرمز.
        """
        if sku not in self._products:
            raise ProductNotFoundError(f"المنتج {sku} غير موجود")
        return self._products[sku]

    def sell(self, sku: str, quantity: int) -> float:
        """بيع كمية من منتج، وإرجاع الإيراد الناتج.

        Raises:
            ProductNotFoundError: إذا لم يوجد المنتج.
            ValueError: إذا كانت الكمية المطلوبة غير موجبة.
            InsufficientStockError: إذا تجاوزت الكمية المطلوبة المخزون المتاح.
        """
        if quantity <= 0:
            raise ValueError("الكمية يجب أن تكون موجبة")

        product = self.get_product(sku)

        if quantity > product.quantity:
            raise InsufficientStockError(
                f"الكمية المتوفرة ({product.quantity}) أقل من المطلوبة ({quantity})"
            )

        product.quantity -= quantity
        return product.price * quantity

    def total_inventory_value(self) -> float:
        """حساب القيمة الإجمالية لكل المخزون الحالي."""
        return sum(p.price * p.quantity for p in self._products.values())

    def low_stock_products(self, threshold: int = 5) -> list[Product]:
        """إرجاع قائمة المنتجات التي كميتها أقل من أو تساوي حدًا معيّنًا."""
        return [p for p in self._products.values() if p.quantity <= threshold]

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
"""Fixtures مشتركة متاحة تلقائيًا لكل ملفات الاختبار في هذا المجلد."""

import pytest

from inventory.inventory_manager import InventoryManager
from inventory.models import Product

@pytest.fixture
def empty_manager():
    """مدير مخزون فارغ تمامًا."""
    return InventoryManager()

@pytest.fixture
def manager_with_products(empty_manager):
    """مدير مخزون مُعبّأ بمنتجات تجريبية جاهزة للاختبارات."""
    empty_manager.add_product(Product(sku="LAP001", name="لابتوب", price=3500, quantity=10))
    empty_manager.add_product(Product(sku="MOU001", name="ماوس", price=50, quantity=100))
    empty_manager.add_product(Product(sku="KEY001", name="لوحة مفاتيح", price=120, quantity=3))
    return empty_manager

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
"""اختبارات شاملة لنظام إدارة المخزون."""

import pytest

from inventory.models import (
    DuplicateSKUError,
    InsufficientStockError,
    Product,
    ProductNotFoundError,
)

class TestAddProduct:
    """اختبارات إضافة منتج جديد."""

    def test_add_product_succeeds(self, empty_manager):
        product = Product(sku="TEST01", name="منتج تجريبي", price=100, quantity=5)
        empty_manager.add_product(product)

        retrieved = empty_manager.get_product("TEST01")
        assert retrieved.name == "منتج تجريبي"

    def test_add_duplicate_sku_raises_error(self, manager_with_products):
        duplicate = Product(sku="LAP001", name="لابتوب آخر", price=4000, quantity=5)

        with pytest.raises(DuplicateSKUError):
            manager_with_products.add_product(duplicate)

class TestGetProduct:
    """اختبارات استرجاع منتج."""

    def test_get_existing_product(self, manager_with_products):
        product = manager_with_products.get_product("MOU001")
        assert product.name == "ماوس"
        assert product.price == 50

    def test_get_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.get_product("NOTEXIST")

class TestSellProduct:
    """اختبارات عملية البيع."""

    def test_sell_reduces_quantity(self, manager_with_products):
        manager_with_products.sell("LAP001", 3)
        remaining = manager_with_products.get_product("LAP001")
        assert remaining.quantity == 7

    def test_sell_returns_correct_revenue(self, manager_with_products):
        revenue = manager_with_products.sell("MOU001", 10)
        assert revenue == 500  # 50 * 10

    def test_sell_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.sell("NOTEXIST", 1)

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -100])
    def test_sell_non_positive_quantity_raises_error(
        self, manager_with_products, invalid_quantity
    ):
        with pytest.raises(ValueError):
            manager_with_products.sell("LAP001", invalid_quantity)

    def test_sell_more_than_available_raises_error(self, manager_with_products):
        with pytest.raises(InsufficientStockError):
            manager_with_products.sell("KEY001", 1000)

    def test_sell_exact_remaining_quantity_succeeds(self, manager_with_products):
        """حالة حدّية: بيع الكمية المتبقية بالضبط يجب أن ينجح (وليس يرفض)."""
        manager_with_products.sell("KEY001", 3)  # الكمية المتوفرة بالضبط
        remaining = manager_with_products.get_product("KEY001")
        assert remaining.quantity == 0

class TestInventoryReports:
    """اختبارات تقارير المخزون الإجمالية."""

    def test_total_inventory_value(self, manager_with_products):
        # (3500*10) + (50*100) + (120*3) = 35000 + 5000 + 360 = 40360
        assert manager_with_products.total_inventory_value() == 40360

    def test_empty_inventory_value_is_zero(self, empty_manager):
        assert empty_manager.total_inventory_value() == 0

    def test_low_stock_products_default_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products()
        skus = [p.sku for p in low_stock]
        assert "KEY001" in skus      # كمية 3، أقل من الحد الافتراضي 5
        assert "LAP001" not in skus  # كمية 10، أعلى من الحد

    def test_low_stock_products_custom_threshold(self, manager_with_products):
        low_stock = manager_with_products.low_stock_products(threshold=15)
        skus = [p.sku for p in low_stock]
        assert "LAP001" in skus  # كمية 10، أقل من الحد المخصص 15

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
# دوال للاختبار (ضعها في ملف منفصل src/utils.py)
def is_palindrome(text: str) -> bool:
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def calculate_average(numbers: list) -> float:
    if not numbers:
        raise ValueError("القائمة لا يمكن أن تكون فارغة")
    return sum(numbers) / len(numbers)

def is_even(n: int) -> bool:
    return n % 2 == 0

# الاختبارات
import pytest
from utils import is_palindrome, calculate_average, is_even

def test_is_palindrome_true_case():
    assert is_palindrome("مدم") is True

def test_is_palindrome_false_case():
    assert is_palindrome("بيت") is False

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("أ") is True

def test_calculate_average_empty_list_raises_error():
    with pytest.raises(ValueError):
        calculate_average([])

@pytest.mark.parametrize("n, expected", [
    (2, True), (3, False), (0, True), (-4, True), (-7, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
# إضافة إلى inventory_manager.py
def restock(self, sku: str, quantity: int) -> None:
    """زيادة كمية منتج موجود في المخزون.

    Raises:
        ProductNotFoundError: إذا لم يوجد المنتج.
        ValueError: إذا كانت الكمية المُضافة غير موجبة.
    """
    if quantity <= 0:
        raise ValueError("كمية التزويد يجب أن تكون موجبة")

    product = self.get_product(sku)  # يرفع ProductNotFoundError تلقائيًا إن لم يوجد
    product.quantity += quantity

# الاختبارات
class TestRestock:
    def test_restock_increases_quantity(self, manager_with_products):
        manager_with_products.restock("KEY001", 10)
        product = manager_with_products.get_product("KEY001")
        assert product.quantity == 13

    @pytest.mark.parametrize("invalid_quantity", [0, -1, -50])
    def test_restock_non_positive_raises_error(self, manager_with_products, invalid_quantity):
        with pytest.raises(ValueError):
            manager_with_products.restock("KEY001", invalid_quantity)

    def test_restock_nonexistent_product_raises_error(self, manager_with_products):
        with pytest.raises(ProductNotFoundError):
            manager_with_products.restock("NOTEXIST", 10)

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-25-chapter-25.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
# ===== الأساسيات =====
def test_something():                    # يجب أن يبدأ بـ test_
    assert actual == expected

# ===== اختبار الاستثناءات =====
import pytest
with pytest.raises(ValueError):
    risky_function()
with pytest.raises(ValueError, match="رسالة محددة"):
    risky_function()

# ===== الأعداد العشرية =====
assert 0.1 + 0.2 == pytest.approx(0.3)

# ===== Fixtures =====
@pytest.fixture
def my_data():
    return {"key": "value"}

def test_uses_fixture(my_data):
    assert my_data["key"] == "value"

@pytest.fixture
def resource_with_cleanup():
    resource = setup()
    yield resource
    teardown(resource)

# ===== Parametrize =====
@pytest.mark.parametrize("a, b, expected", [(1,1,2), (2,2,4)])
def test_add(a, b, expected):
    assert add(a, b) == expected

# ===== Mocking =====
from unittest.mock import patch, MagicMock
with patch("module.function_name") as mock_func:
    mock_func.return_value = "قيمة وهمية"

# ===== أوامر التشغيل =====
# uv run pytest                              تشغيل كل الاختبارات
# uv run pytest -v                           تفصيلي
# uv run pytest tests/test_x.py::test_y      اختبار محدد
# uv run pytest --cov=src                    مع تقرير التغطية
# uv run pytest -k "keyword"                 فلترة بالاسم
