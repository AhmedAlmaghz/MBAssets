#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
Python 3.14 — From Zero to Pro — Type Hints

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 96 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-15-chapter-15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
Python 3.14 — From Zero to Pro — Type Hints

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-15-chapter-15 (lesson 15)
Code files: 95 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["mypy"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q mypy

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price, percentage, customer):
    ...

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
age: int = 25
name: str = "محمد"

def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# متغيرات
age: int = 25
price: float = 19.99
name: str = "سارة"
is_active: bool = True

# دوال
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"مرحبًا، {name}")

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")  # لا يوجد أي خطأ من Python!
print(result)  # "مرحبًا!"

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# قائمة أعداد صحيحة
numbers: list[int] = [1, 2, 3, 4, 5]

# قائمة نصوص
names: list[str] = ["أحمد", "سارة", "خالد"]

# قائمة متداخلة (قائمة من قوائم أعداد)
matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# صندوق يحتوي اسمًا (نص) وعمرًا (عدد صحيح)
person: tuple[str, int] = ("ليلى", 28)

# صندوق إحداثيات (x, y, z) - ثلاثة أعداد عشرية
point: tuple[float, float, float] = (1.5, 2.0, 3.5)

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# صندوق بعدد غير محدد من الأعداد الصحيحة
scores: tuple[int, ...] = (90, 85, 100, 70, 60)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# قاموس: المفتاح نص، القيمة عدد صحيح
ages: dict[str, int] = {"أحمد": 30, "سارة": 25}

# قاموس أكثر تعقيدًا: المفتاح نص، القيمة قائمة نصوص
courses_by_student: dict[str, list[str]] = {
    "أحمد": ["رياضيات", "فيزياء"],
    "سارة": ["كيمياء"],
}

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
unique_ids: set[int] = {101, 102, 103}
tags: set[str] = {"python", "برمجة", "تعليم"}

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def summarize_students(
    grades: dict[str, list[int]]
) -> dict[str, float]:
    """
    يستقبل قاموسًا بأسماء الطلاب ودرجاتهم،
    ويُرجع قاموسًا بأسماء الطلاب ومتوسط درجاتهم.
    """
    averages: dict[str, float] = {}
    for student, scores in grades.items():
        averages[student] = sum(scores) / len(scores)
    return averages

student_grades: dict[str, list[int]] = {
    "محمد": [85, 90, 78],
    "فاطمة": [92, 88, 95],
}

result: dict[str, float] = summarize_students(student_grades)
print(result)
# {'محمد': 84.33333333333333, 'فاطمة': 91.66666666666667}

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-15.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def find_user(user_id: int, users: dict[int, str]) -> str:
    return users.get(user_id)  # قد تُرجع None إن لم يوجد المستخدم!

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-15.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from typing import Optional

def find_user_old_style(user_id: int, users: dict[int, str]) -> Optional[str]:
    return users.get(user_id)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-15.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

result: str | None = find_user(5, {1: "أحمد", 2: "سارة"})

if result is not None:
    print(result.upper())  # آمن تمامًا، لأننا تحققنا أولًا
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-15.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# معامل اختياري (له قيمة افتراضية) - لا علاقة له بـ Optional type
def greet(name: str = "ضيف") -> str:
    return f"مرحبًا، {name}"

# Optional type - القيمة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-15.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def greet(name: str | None = None) -> str:
    if name is None:
        name = "ضيف"
    return f"مرحبًا، {name}"

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-15.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def format_id(value: int | str) -> str:
    """يحوّل معرّفًا (رقمًا أو نصًا) إلى نص منسّق."""
    return f"ID-{value}"

print(format_id(42))      # "ID-42"
print(format_id("A001"))  # "ID-A001"

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-15.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
from typing import Union

def format_id(value: Union[int, str]) -> str:
    return f"ID-{value}"

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-15.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def process(value: int | str) -> None:
    if isinstance(value, int):
        print(f"رقم مضاعف: {value * 2}")
    else:
        print(f"نص بأحرف كبيرة: {value.upper()}")

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-15.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
from typing import Any

def process_anything(data: Any) -> Any:
    return data

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-15.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
def func_any(x: Any) -> None:
    x.anything_at_all()  # لا تحذير من mypy حتى لو الدالة غير موجودة!

def func_object(x: object) -> None:
    x.anything_at_all()  # mypy يُصدر خطأً! object لا يملك هذه الدالة

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-15.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
from typing import Callable

def apply_operation(
    numbers: list[int],
    operation: Callable[[int], int]
) -> list[int]:
    """يطبّق 'operation' على كل عنصر في القائمة."""
    return [operation(n) for n in numbers]

def double(x: int) -> int:
    return x * 2

def square(x: int) -> int:
    return x * x

print(apply_operation([1, 2, 3], double))  # [2, 4, 6]
print(apply_operation([1, 2, 3], square))  # [1, 4, 9]

# يعمل أيضًا مع lambda
print(apply_operation([1, 2, 3], lambda x: x + 10))  # [11, 12, 13]

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-15.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# دالة بدون معاملات تُرجع نصًا
greeting_provider: Callable[[], str]

# دالة تستقبل int و str وتُرجع bool
validator: Callable[[int, str], bool]

# دالة بأي عدد ونوع من المعاملات (تجنّب هذا إن أمكن، لكنه موجود)
from typing import Callable
any_callable: Callable[..., int]  # ... تعني "أي معاملات"

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-15.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from typing import Callable
import time
import functools

def measure_time(func: Callable[..., float]) -> Callable[..., float]:
    @functools.wraps(func)
    def wrapper(*args: int, **kwargs: int) -> float:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"استغرقت {func.__name__}: {elapsed:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> float:
    return sum(i ** 2 for i in range(n)) / n

slow_calculation(1_000_000)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-15.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def get_first_element(items: list) -> Any:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-15.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
from typing import TypeVar

T = TypeVar("T")

def get_first_element(items: list[T]) -> T:
    return items[0]

numbers: list[int] = [10, 20, 30]
first_number = get_first_element(numbers)  # mypy يستنتج: int

words: list[str] = ["كتاب", "قلم"]
first_word = get_first_element(words)  # mypy يستنتج: str

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-15.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def get_first_element[T](items: list[T]) -> T:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-15.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
class Box[T]:
    """صندوق يحتفظ بعنصر واحد من أي نوع."""

    def __init__(self, content: T) -> None:
        self.content = content

    def get(self) -> T:
        return self.content

    def __repr__(self) -> str:
        return f"Box({self.content!r})"

int_box = Box(42)           # mypy يستنتج: Box[int]
str_box = Box("مرحبًا")      # mypy يستنتج: Box[str]

print(int_box.get())  # 42
print(str_box.get())  # مرحبًا

value: int = int_box.get()  # صحيح، mypy يعرف أن الناتج int

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-15.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
class Repository[T]:
    """مخزن بسيط يدير مجموعة من العناصر من نوع T."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        return self._items.copy()

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

from typing import Callable
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product_repo = Repository[Product]()
product_repo.add(Product("لابتوب", 1200.0))
product_repo.add(Product("هاتف", 800.0))

cheap_product = product_repo.find(lambda p: p.price < 1000)
print(cheap_product)  # Product(name='هاتف', price=800.0)

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-15.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "رسم دائرة ⭕"

class Square:
    def draw(self) -> str:
        return "رسم مربع ⬜"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # صحيح! Circle تملك draw()
render(Square())  # صحيح! Square تملك draw() أيضًا

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-15.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-15.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
user = find_user(5, {1: "أحمد"})
if user is not None:
    print(user.upper())
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [033] dist/notebooks/chapter-15.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
# ✅ جيد: التركيز على توقيع الدالة (المعاملات والإرجاع)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)  # mypy يستنتج أن subtotal من نوع float
    return subtotal * (1 + tax_rate)

# ────────────────────────────────────────────────────────────────────────
# [034] dist/notebooks/chapter-15.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
"""
نظام إدارة المهام - مثال تطبيقي شامل على Type Hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, Callable

# ---------------------------------------------------------
# 1. تعريف الأنواع الأساسية باستخدام Literal و Type Alias
# ---------------------------------------------------------
type Priority = Literal["منخفضة", "متوسطة", "عالية"]
type Status = Literal["معلّقة", "قيد التنفيذ", "منجزة"]

# ---------------------------------------------------------
# 2. Protocol لأي شيء "قابل للعرض"
# ---------------------------------------------------------
class Displayable(Protocol):
    def display(self) -> str:
        ...

# ---------------------------------------------------------
# 3. فئة المهمة (Task) باستخدام dataclass
# ---------------------------------------------------------
@dataclass
class Task:
    title: str
    priority: Priority = "متوسطة"
    status: Status = "معلّقة"
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def display(self) -> str:
        due_info = f" (موعد التسليم: {self.due_date})" if self.due_date else ""
        tags_info = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"📋 {self.title} - {self.priority} - {self.status}{due_info}{tags_info}"

    def mark_as(self, new_status: Status) -> None:
        self.status = new_status

# ---------------------------------------------------------
# 4. مستودع عام (Generic Repository) لإدارة أي نوع من العناصر
# ---------------------------------------------------------
class Repository[T]:
    """مستودع عام يدير مجموعة من العناصر من أي نوع."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def remove(self, item: T) -> None:
        self._items.remove(item)

    def all(self) -> list[T]:
        return self._items.copy()

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        return [item for item in self._items if predicate(item)]

    def find_first(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(self._items)

# ---------------------------------------------------------
# 5. مدير المهام (TaskManager) - يستخدم Repository
# ---------------------------------------------------------
class TaskManager:
    def __init__(self) -> None:
        self._repo: Repository[Task] = Repository()

    def add_task(
        self,
        title: str,
        priority: Priority = "متوسطة",
        due_date: date | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task = Task(
            title=title,
            priority=priority,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        self._repo.add(task)
        return task

    def get_by_status(self, status: Status) -> list[Task]:
        return self._repo.filter(lambda t: t.status == status)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        return self._repo.filter(lambda t: t.priority == priority)

    def find_task_by_title(self, title: str) -> Task | None:
        return self._repo.find_first(lambda t: t.title == title)

    def complete_task(self, title: str) -> bool:
        task = self.find_task_by_title(title)
        if task is None:
            return False
        task.mark_as("منجزة")
        return True

    def summary(self) -> dict[Status, int]:
        statuses: list[Status] = ["معلّقة", "قيد التنفيذ", "منجزة"]
        return {
            status: len(self.get_by_status(status))
            for status in statuses
        }

    def display_all(self) -> None:
        if self._repo.count() == 0:
            print("لا توجد مهام بعد.")
            return
        for task in self._repo.all():
            print(task.display())

# ---------------------------------------------------------
# 6. دالة عامة تعمل مع أي كائن Displayable (Protocol)
# ---------------------------------------------------------
def print_report(items: list[Displayable], title: str = "تقرير") -> None:
    print(f"=== {title} ===")
    for item in items:
        print(item.display())
    print(f"عدد العناصر: {len(items)}")

# ---------------------------------------------------------
# 7. برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = TaskManager()

    manager.add_task(
        "كتابة تقرير شهري",
        priority="عالية",
        due_date=date(2025, 6, 30),
        tags=["عمل", "عاجل"],
    )
    manager.add_task("قراءة كتاب Python", priority="منخفضة")
    manager.add_task(
        "مراجعة الكود مع الفريق",
        priority="متوسطة",
        tags=["فريق"],
    )

    print("كل المهام:")
    manager.display_all()
    print()

    manager.complete_task("قراءة كتاب Python")

    print("المهام العالية الأولوية:")
    for task in manager.get_by_priority("عالية"):
        print(task.display())
    print()

    print("ملخص الحالات:")
    for status, count in manager.summary().items():
        print(f"  {status}: {count}")
    print()

    high_priority_tasks: list[Displayable] = list(manager.get_by_priority("عالية"))
    print_report(high_priority_tasks, title="تقرير المهام العاجلة")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [035] dist/notebooks/chapter-15.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def calculate_area(width, height):
    return width * height

# ────────────────────────────────────────────────────────────────────────
# [036] dist/notebooks/chapter-15.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def calculate_area(width: float, height: float) -> float:
    return width * height

# تمرين 2
from typing import Literal

Grade = Literal["ممتاز", "جيد", "راسب"]

def get_grade(score: int) -> Grade:
    if score >= 90:
        return "ممتاز"
    elif score >= 60:
        return "جيد"
    else:
        return "راسب"

# تمرين 3
# لا فرق في المعنى: كلاهما يعني "قد يكون int أو None".
# Optional[int] هي الصيغة القديمة (تحتاج: from typing import Optional).
# int | None هي الصيغة الحديثة الموصى بها في Python 3.10+ (لا تحتاج استيراد).
# في كتابنا نستخدم دائمًا: int | None

# ────────────────────────────────────────────────────────────────────────
# [037] dist/notebooks/chapter-15.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def find_product(product_id: int, catalog: dict[int, str]) -> str:
        return catalog.get(product_id)

    def get_discount(price: float, percent: int) -> float:
        return price - (price * percent)

    products = {1: "لابتوب", 2: "هاتف"}
    name = find_product(3, products)
    print(name.upper())
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [038] dist/notebooks/chapter-15.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # تمرين 4 - الكود المصحح
    def find_product(product_id: int, catalog: dict[int, str]) -> str | None:
        return catalog.get(product_id)  # قد تكون None فعلاً

    def get_discount(price: float, percent: int) -> float:
        return price - (price * (percent / 100))  # كانت تُطرح قيمة كبيرة جدًا خطأً منطقيًا

    products: dict[int, str] = {1: "لابتوب", 2: "هاتف"}
    name = find_product(3, products)

    if name is not None:
        print(name.upper())
    else:
        print("المنتج غير موجود")

    # تمرين 5
    class Stack[T]:
        def __init__(self) -> None:
            self._items: list[T] = []

        def push(self, item: T) -> None:
            self._items.append(item)

        def pop(self) -> T | None:
            if self.is_empty():
                return None
            return self._items.pop()

        def peek(self) -> T | None:
            if self.is_empty():
                return None
            return self._items[-1]

        def is_empty(self) -> bool:
            return len(self._items) == 0

    stack: Stack[int] = Stack()
    stack.push(1)
    stack.push(2)
    print(stack.pop())   # 2
    print(stack.peek())  # 1

    # تمرين 6
    from typing import Protocol

    class Comparable(Protocol):
        def __lt__(self, other: object) -> bool:
            ...

    def find_max[T: Comparable](items: list[T]) -> T:
        if not items:
            raise ValueError("القائمة فارغة")
        current_max = items[0]
        for item in items[1:]:
            if current_max < item:
                current_max = item
        return current_max

    print(find_max([3, 7, 2, 9, 4]))        # 9
    print(find_max(["بطة", "قطة", "أسد"]))  # قطة (حسب الترتيب الأبجدي)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [039] dist/notebooks/chapter-15.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
# ===== الأنواع الأساسية =====
age: int = 25
price: float = 19.99
name: str = "سارة"
active: bool = True

# ===== الهياكل المركّبة =====
numbers: list[int] = [1, 2, 3]
point: tuple[float, float] = (1.0, 2.0)
scores: tuple[int, ...] = (1, 2, 3, 4)
ages: dict[str, int] = {"أحمد": 30}
tags: set[str] = {"a", "b"}

# ===== القيم الاختيارية والمركّبة =====
result: str | None = None          # اختياري (الصيغة الحديثة)
value: int | str = 5               # إمّا هذا أو ذاك

# ===== Any (استخدمها بحذر!) =====
from typing import Any
data: Any = {"whatever": "shape"}

# ===== Callable (الدوال كقيم) =====
from typing import Callable
operation: Callable[[int, int], int]

# ===== Literal (قيم محددة فقط) =====
from typing import Literal
Status = Literal["pending", "done"]

# ===== TypedDict (قاموس مُهيكَل) =====
from typing import TypedDict
class User(TypedDict):
    name: str
    age: int

# ===== Generics (البرمجة العامة) =====
def first[T](items: list[T]) -> T:
    return items[0]

class Box[T]:
    def __init__(self, content: T) -> None:
        self.content = content

# ===== Protocol (duck typing مُهيكَل) =====
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> str: ...

# ===== mypy =====
# التثبيت:
#   uv add --dev mypy
# التشغيل:
#   uv run mypy file.py
#   uv run mypy . --strict
# الإعداد في pyproject.toml:
#   [tool.mypy]
#   strict = true

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch15/15-001-calculate_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-001-calculate_discount.py
"""

def calculate_discount(price, percentage, customer):
    ...

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch15/15-002-calculate_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-002-calculate_discount.py
"""

def calculate_discount(price: float, percentage: float, customer: Customer) -> float:
    ...

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch15/15-003-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-003-greet.py
"""

age: int = 25
name: str = "محمد"

def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch15/15-004-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-004-add.py
"""

# متغيرات
age: int = 25
price: float = 19.99
name: str = "سارة"
is_active: bool = True

# دوال
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"مرحبًا، {name}")

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch15/15-005-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-005-add.py
"""

def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")  # لا يوجد أي خطأ من Python!
print(result)  # "مرحبًا!"

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch15/15-006-ex15_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-006-ex15_3.py
"""

# قائمة أعداد صحيحة
numbers: list[int] = [1, 2, 3, 4, 5]

# قائمة نصوص
names: list[str] = ["أحمد", "سارة", "خالد"]

# قائمة متداخلة (قائمة من قوائم أعداد)
matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch15/15-007-ex15_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-007-ex15_3.py
"""

# صندوق يحتوي اسمًا (نص) وعمرًا (عدد صحيح)
person: tuple[str, int] = ("ليلى", 28)

# صندوق إحداثيات (x, y, z) - ثلاثة أعداد عشرية
point: tuple[float, float, float] = (1.5, 2.0, 3.5)

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch15/15-008-ex15_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-008-ex15_3.py
"""

# صندوق بعدد غير محدد من الأعداد الصحيحة
scores: tuple[int, ...] = (90, 85, 100, 70, 60)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch15/15-009-ex15_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-009-ex15_3.py
"""

# قاموس: المفتاح نص، القيمة عدد صحيح
ages: dict[str, int] = {"أحمد": 30, "سارة": 25}

# قاموس أكثر تعقيدًا: المفتاح نص، القيمة قائمة نصوص
courses_by_student: dict[str, list[str]] = {
    "أحمد": ["رياضيات", "فيزياء"],
    "سارة": ["كيمياء"],
}

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch15/15-010-ex15_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-010-ex15_3.py
"""

unique_ids: set[int] = {101, 102, 103}
tags: set[str] = {"python", "برمجة", "تعليم"}

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch15/15-011-summarize_students.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-011-summarize_students.py
"""

def summarize_students(
    grades: dict[str, list[int]]
) -> dict[str, float]:
    """
    يستقبل قاموسًا بأسماء الطلاب ودرجاتهم،
    ويُرجع قاموسًا بأسماء الطلاب ومتوسط درجاتهم.
    """
    averages: dict[str, float] = {}
    for student, scores in grades.items():
        averages[student] = sum(scores) / len(scores)
    return averages

student_grades: dict[str, list[int]] = {
    "محمد": [85, 90, 78],
    "فاطمة": [92, 88, 95],
}

result: dict[str, float] = summarize_students(student_grades)
print(result)
# {'محمد': 84.33333333333333, 'فاطمة': 91.66666666666667}

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch15/15-012-find_user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-012-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str:
    return users.get(user_id)  # قد تُرجع None إن لم يوجد المستخدم!

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch15/15-013-find_user_old_style.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-013-find_user_old_style.py
"""

from typing import Optional

def find_user_old_style(user_id: int, users: dict[int, str]) -> Optional[str]:
    return users.get(user_id)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch15/15-014-find_user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-014-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

result: str | None = find_user(5, {1: "أحمد", 2: "سارة"})

if result is not None:
    print(result.upper())  # آمن تمامًا، لأننا تحققنا أولًا
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch15/15-015-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-015-greet.py
"""

# معامل اختياري (له قيمة افتراضية) - لا علاقة له بـ Optional type
def greet(name: str = "ضيف") -> str:
    return f"مرحبًا، {name}"

# Optional type - القيمة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch15/15-016-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-016-greet.py
"""

def greet(name: str | None = None) -> str:
    if name is None:
        name = "ضيف"
    return f"مرحبًا، {name}"

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch15/15-017-format_id.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-017-format_id.py
"""

def format_id(value: int | str) -> str:
    """يحوّل معرّفًا (رقمًا أو نصًا) إلى نص منسّق."""
    return f"ID-{value}"

print(format_id(42))      # "ID-42"
print(format_id("A001"))  # "ID-A001"

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch15/15-018-format_id.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-018-format_id.py
"""

from typing import Union

def format_id(value: Union[int, str]) -> str:
    return f"ID-{value}"

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch15/15-019-process.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-019-process.py
"""

def process(value: int | str) -> None:
    if isinstance(value, int):
        print(f"رقم مضاعف: {value * 2}")
    else:
        print(f"نص بأحرف كبيرة: {value.upper()}")

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch15/15-020-process_anything.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-020-process_anything.py
"""

from typing import Any

def process_anything(data: Any) -> Any:
    return data

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch15/15-021-func_any.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-021-func_any.py
"""

def func_any(x: Any) -> None:
    x.anything_at_all()  # لا تحذير من mypy حتى لو الدالة غير موجودة!

def func_object(x: object) -> None:
    x.anything_at_all()  # mypy يُصدر خطأً! object لا يملك هذه الدالة

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch15/15-022-apply_operation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-022-apply_operation.py
"""

from typing import Callable

def apply_operation(
    numbers: list[int],
    operation: Callable[[int], int]
) -> list[int]:
    """يطبّق 'operation' على كل عنصر في القائمة."""
    return [operation(n) for n in numbers]

def double(x: int) -> int:
    return x * 2

def square(x: int) -> int:
    return x * x

print(apply_operation([1, 2, 3], double))  # [2, 4, 6]
print(apply_operation([1, 2, 3], square))  # [1, 4, 9]

# يعمل أيضًا مع lambda
print(apply_operation([1, 2, 3], lambda x: x + 10))  # [11, 12, 13]

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch15/15-023-ex15_7.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-023-ex15_7.py
"""

# دالة بدون معاملات تُرجع نصًا
greeting_provider: Callable[[], str]

# دالة تستقبل int و str وتُرجع bool
validator: Callable[[int, str], bool]

# دالة بأي عدد ونوع من المعاملات (تجنّب هذا إن أمكن، لكنه موجود)
from typing import Callable
any_callable: Callable[..., int]  # ... تعني "أي معاملات"

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch15/15-024-measure_time.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-024-measure_time.py
"""

from typing import Callable
import time
import functools

def measure_time(func: Callable[..., float]) -> Callable[..., float]:
    @functools.wraps(func)
    def wrapper(*args: int, **kwargs: int) -> float:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"استغرقت {func.__name__}: {elapsed:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> float:
    return sum(i ** 2 for i in range(n)) / n

slow_calculation(1_000_000)

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch15/15-025-set_status.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-025-set_status.py
"""

from typing import Literal

def set_status(status: Literal["active", "inactive", "pending"]) -> None:
    print(f"الحالة الجديدة: {status}")

set_status("active")    # صحيح
set_status("pending")   # صحيح
set_status("deleted")   # ❌ mypy يرفض هذا! "deleted" ليست من القيم المسموحة

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch15/15-026-move.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-026-move.py
"""

Direction = Literal["up", "down", "left", "right"]

def move(direction: Direction, steps: int = 1) -> tuple[int, int]:
    deltas: dict[Direction, tuple[int, int]] = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    dx, dy = deltas[direction]
    return (dx * steps, dy * steps)

print(move("up", 3))  # (0, 3)

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch15/15-027-print_employee.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.9: القواميس المُهيكَلة: TypedDict

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-027-print_employee.py
"""

from typing import TypedDict

class Employee(TypedDict):
    name: str
    age: int
    salary: float

def print_employee(emp: Employee) -> None:
    print(f"{emp['name']} - {emp['age']} سنة - راتب {emp['salary']} دولار")

employee1: Employee = {
    "name": "ياسر",
    "age": 35,
    "salary": 5000.0,
}

print_employee(employee1)

# استخدام خاطئ يكتشفه mypy:
bad_employee: Employee = {
    "name": "سلمى",
    "age": "ثلاثون",  # ❌ خطأ! يجب أن يكون int وليس str
    "salary": 4500.0,
}

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch15/15-028-get_first_element.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-028-get_first_element.py
"""

def get_first_element(items: list) -> Any:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch15/15-029-get_first_element.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-029-get_first_element.py
"""

from typing import TypeVar

T = TypeVar("T")

def get_first_element(items: list[T]) -> T:
    return items[0]

numbers: list[int] = [10, 20, 30]
first_number = get_first_element(numbers)  # mypy يستنتج: int

words: list[str] = ["كتاب", "قلم"]
first_word = get_first_element(words)  # mypy يستنتج: str

# ────────────────────────────────────────────────────────────────────────
# [069] code/chapters/ch15/15-030-get_first_element.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-030-get_first_element.py
"""

def get_first_element[T](items: list[T]) -> T:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [070] code/chapters/ch15/15-031-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-031-__init__.py
"""

class Box[T]:
    """صندوق يحتفظ بعنصر واحد من أي نوع."""

    def __init__(self, content: T) -> None:
        self.content = content

    def get(self) -> T:
        return self.content

    def __repr__(self) -> str:
        return f"Box({self.content!r})"

int_box = Box(42)           # mypy يستنتج: Box[int]
str_box = Box("مرحبًا")      # mypy يستنتج: Box[str]

print(int_box.get())  # 42
print(str_box.get())  # مرحبًا

value: int = int_box.get()  # صحيح، mypy يعرف أن الناتج int

# ────────────────────────────────────────────────────────────────────────
# [071] code/chapters/ch15/15-032-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-032-__init__.py
"""

class Repository[T]:
    """مخزن بسيط يدير مجموعة من العناصر من نوع T."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        return self._items.copy()

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

from typing import Callable
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product_repo = Repository[Product]()
product_repo.add(Product("لابتوب", 1200.0))
product_repo.add(Product("هاتف", 800.0))

cheap_product = product_repo.find(lambda p: p.price < 1000)
print(cheap_product)  # Product(name='هاتف', price=800.0)

# ────────────────────────────────────────────────────────────────────────
# [072] code/chapters/ch15/15-033-draw.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.11: البروتوكولات: Protocol (Duck Typing المُهيكَل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-033-draw.py
"""

from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "رسم دائرة ⭕"

class Square:
    def draw(self) -> str:
        return "رسم مربع ⬜"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # صحيح! Circle تملك draw()
render(Square())  # صحيح! Square تملك draw() أيضًا

# ────────────────────────────────────────────────────────────────────────
# [073] code/chapters/ch15/15-034-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-034-add.py
"""

def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [074] code/chapters/ch15/15-035-find_user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-035-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

user = find_user(5, {1: "أحمد"})
print(user.upper())  # ❌ خطأ محتمل!

# ────────────────────────────────────────────────────────────────────────
# [075] code/chapters/ch15/15-036-user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-036-user.py
"""

user = find_user(5, {1: "أحمد"})
if user is not None:
    print(user.upper())
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [076] code/chapters/ch15/15-037-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-037-greet.py
"""

def greet(name: str, greeting: str = "مرحبًا") -> str:
    return f"{greeting}، {name}!"

print(greet())  # ❌ نسينا معاملًا إلزاميًا

# ────────────────────────────────────────────────────────────────────────
# [077] code/chapters/ch15/15-038-get_age.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-038-get_age.py
"""

def get_age() -> int:
    return "عشرون"  # ❌ نص بدل عدد صحيح

# ────────────────────────────────────────────────────────────────────────
# [078] code/chapters/ch15/15-039-point.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-039-point.py
"""

from typing import TypedDict

class Point(TypedDict):
    x: int
    y: int

p: Point = {"x": 1}  # ❌ ناقص حقل y

# ────────────────────────────────────────────────────────────────────────
# [079] code/chapters/ch15/15-040-result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-040-result.py
"""

import some_untyped_library  # type: ignore[import-untyped]

result = some_untyped_library.do_something()

# ────────────────────────────────────────────────────────────────────────
# [080] code/chapters/ch15/15-041-calculate_total.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-041-calculate_total.py
"""

# ✅ جيد: التركيز على توقيع الدالة (المعاملات والإرجاع)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)  # mypy يستنتج أن subtotal من نوع float
    return subtotal * (1 + tax_rate)

# ────────────────────────────────────────────────────────────────────────
# [081] code/chapters/ch15/15-042-find.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-042-find.py
"""

# ❌ غامض: هل يمكن أن يكون None؟ لا نعرف
def find(key: str) -> str:
    ...

# ✅ واضح: نعرف بالضبط أن النتيجة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [082] code/chapters/ch15/15-043-ex15_13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-043-ex15_13.py
"""

x = [1, 2, 3]
reveal_type(x)  # أداة تشخيصية خاصة بـ mypy فقط

# ────────────────────────────────────────────────────────────────────────
# [083] code/chapters/ch15/15-044-display.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.14: مشروع تطبيقي: نظام إدارة مهام مُنمَّط بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-044-display.py
"""

"""
نظام إدارة المهام - مثال تطبيقي شامل على Type Hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, Callable

# ---------------------------------------------------------
# 1. تعريف الأنواع الأساسية باستخدام Literal و Type Alias
# ---------------------------------------------------------
type Priority = Literal["منخفضة", "متوسطة", "عالية"]
type Status = Literal["معلّقة", "قيد التنفيذ", "منجزة"]

# ---------------------------------------------------------
# 2. Protocol لأي شيء "قابل للعرض"
# ---------------------------------------------------------
class Displayable(Protocol):
    def display(self) -> str:
        ...

# ---------------------------------------------------------
# 3. فئة المهمة (Task) باستخدام dataclass
# ---------------------------------------------------------
@dataclass
class Task:
    title: str
    priority: Priority = "متوسطة"
    status: Status = "معلّقة"
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def display(self) -> str:
        due_info = f" (موعد التسليم: {self.due_date})" if self.due_date else ""
        tags_info = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"📋 {self.title} - {self.priority} - {self.status}{due_info}{tags_info}"

    def mark_as(self, new_status: Status) -> None:
        self.status = new_status

# ---------------------------------------------------------
# 4. مستودع عام (Generic Repository) لإدارة أي نوع من العناصر
# ---------------------------------------------------------
class Repository[T]:
    """مستودع عام يدير مجموعة من العناصر من أي نوع."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def remove(self, item: T) -> None:
        self._items.remove(item)

    def all(self) -> list[T]:
        return self._items.copy()

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        return [item for item in self._items if predicate(item)]

    def find_first(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(self._items)

# ---------------------------------------------------------
# 5. مدير المهام (TaskManager) - يستخدم Repository
# ---------------------------------------------------------
class TaskManager:
    def __init__(self) -> None:
        self._repo: Repository[Task] = Repository()

    def add_task(
        self,
        title: str,
        priority: Priority = "متوسطة",
        due_date: date | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task = Task(
            title=title,
            priority=priority,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        self._repo.add(task)
        return task

    def get_by_status(self, status: Status) -> list[Task]:
        return self._repo.filter(lambda t: t.status == status)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        return self._repo.filter(lambda t: t.priority == priority)

    def find_task_by_title(self, title: str) -> Task | None:
        return self._repo.find_first(lambda t: t.title == title)

    def complete_task(self, title: str) -> bool:
        task = self.find_task_by_title(title)
        if task is None:
            return False
        task.mark_as("منجزة")
        return True

    def summary(self) -> dict[Status, int]:
        statuses: list[Status] = ["معلّقة", "قيد التنفيذ", "منجزة"]
        return {
            status: len(self.get_by_status(status))
            for status in statuses
        }

    def display_all(self) -> None:
        if self._repo.count() == 0:
            print("لا توجد مهام بعد.")
            return
        for task in self._repo.all():
            print(task.display())

# ---------------------------------------------------------
# 6. دالة عامة تعمل مع أي كائن Displayable (Protocol)
# ---------------------------------------------------------
def print_report(items: list[Displayable], title: str = "تقرير") -> None:
    print(f"=== {title} ===")
    for item in items:
        print(item.display())
    print(f"عدد العناصر: {len(items)}")

# ---------------------------------------------------------
# 7. برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = TaskManager()

    manager.add_task(
        "كتابة تقرير شهري",
        priority="عالية",
        due_date=date(2025, 6, 30),
        tags=["عمل", "عاجل"],
    )
    manager.add_task("قراءة كتاب Python", priority="منخفضة")
    manager.add_task(
        "مراجعة الكود مع الفريق",
        priority="متوسطة",
        tags=["فريق"],
    )

    print("كل المهام:")
    manager.display_all()
    print()

    manager.complete_task("قراءة كتاب Python")

    print("المهام العالية الأولوية:")
    for task in manager.get_by_priority("عالية"):
        print(task.display())
    print()

    print("ملخص الحالات:")
    for status, count in manager.summary().items():
        print(f"  {status}: {count}")
    print()

    high_priority_tasks: list[Displayable] = list(manager.get_by_priority("عالية"))
    print_report(high_priority_tasks, title="تقرير المهام العاجلة")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [084] code/chapters/ch15/15-045-calculate_area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-045-calculate_area.py
"""

def calculate_area(width, height):
    return width * height

# ────────────────────────────────────────────────────────────────────────
# [085] code/chapters/ch15/15-046-calculate_area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-046-calculate_area.py
"""

# تمرين 1
def calculate_area(width: float, height: float) -> float:
    return width * height

# تمرين 2
from typing import Literal

Grade = Literal["ممتاز", "جيد", "راسب"]

def get_grade(score: int) -> Grade:
    if score >= 90:
        return "ممتاز"
    elif score >= 60:
        return "جيد"
    else:
        return "راسب"

# تمرين 3
# لا فرق في المعنى: كلاهما يعني "قد يكون int أو None".
# Optional[int] هي الصيغة القديمة (تحتاج: from typing import Optional).
# int | None هي الصيغة الحديثة الموصى بها في Python 3.10+ (لا تحتاج استيراد).
# في كتابنا نستخدم دائمًا: int | None

# ────────────────────────────────────────────────────────────────────────
# [086] code/chapters/ch15/15-047-find_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-047-find_product.py
"""

def find_product(product_id: int, catalog: dict[int, str]) -> str:
    return catalog.get(product_id)

def get_discount(price: float, percent: int) -> float:
    return price - (price * percent)

products = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)
print(name.upper())

# ────────────────────────────────────────────────────────────────────────
# [087] code/chapters/ch15/15-048-find_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-048-find_product.py
"""

# تمرين 4 - الكود المصحح
def find_product(product_id: int, catalog: dict[int, str]) -> str | None:
    return catalog.get(product_id)  # قد تكون None فعلاً

def get_discount(price: float, percent: int) -> float:
    return price - (price * (percent / 100))  # كانت تُطرح قيمة كبيرة جدًا خطأً منطقيًا

products: dict[int, str] = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)

if name is not None:
    print(name.upper())
else:
    print("المنتج غير موجود")

# تمرين 5
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T | None:
        if self.is_empty():
            return None
        return self._items.pop()

    def peek(self) -> T | None:
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())   # 2
print(stack.peek())  # 1

# تمرين 6
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other: object) -> bool:
        ...

def find_max[T: Comparable](items: list[T]) -> T:
    if not items:
        raise ValueError("القائمة فارغة")
    current_max = items[0]
    for item in items[1:]:
        if current_max < item:
            current_max = item
    return current_max

print(find_max([3, 7, 2, 9, 4]))        # 9
print(find_max(["بطة", "قطة", "أسد"]))  # قطة (حسب الترتيب الأبجدي)

# ────────────────────────────────────────────────────────────────────────
# [088] code/chapters/ch15/15-049-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-049-__init__.py
"""

from typing import TypedDict, Callable

class CacheStats(TypedDict):
    hits: int
    misses: int

class Cache[K, V]:
    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: K) -> V | None:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get_or_compute(self, key: K, compute_fn: Callable[[], V]) -> V:
        existing = self.get(key)
        if existing is not None:
            return existing
        new_value = compute_fn()
        self.set(key, new_value)
        return new_value

    def stats(self) -> CacheStats:
        return {"hits": self._hits, "misses": self._misses}

def expensive_computation(n: int) -> int:
    print(f"🔄 جارِ الحساب الباهظ لـ {n} ...")
    return n ** 2

cache: Cache[int, int] = Cache()

result1 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result1)  # يطبع "جارِ الحساب..." ثم 25

result2 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result2)  # لا يطبع "جارِ الحساب" لأنها مخزّنة! يطبع 25 مباشرة

print(cache.stats())  # {'hits': 1, 'misses': 1}

# ────────────────────────────────────────────────────────────────────────
# [089] code/chapters/ch15/15-050-first.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.17: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-050-first.py
"""

# ===== الأنواع الأساسية =====
age: int = 25
price: float = 19.99
name: str = "سارة"
active: bool = True

# ===== الهياكل المركّبة =====
numbers: list[int] = [1, 2, 3]
point: tuple[float, float] = (1.0, 2.0)
scores: tuple[int, ...] = (1, 2, 3, 4)
ages: dict[str, int] = {"أحمد": 30}
tags: set[str] = {"a", "b"}

# ===== القيم الاختيارية والمركّبة =====
result: str | None = None          # اختياري (الصيغة الحديثة)
value: int | str = 5               # إمّا هذا أو ذاك

# ===== Any (استخدمها بحذر!) =====
from typing import Any
data: Any = {"whatever": "shape"}

# ===== Callable (الدوال كقيم) =====
from typing import Callable
operation: Callable[[int, int], int]

# ===== Literal (قيم محددة فقط) =====
from typing import Literal
Status = Literal["pending", "done"]

# ===== TypedDict (قاموس مُهيكَل) =====
from typing import TypedDict
class User(TypedDict):
    name: str
    age: int

# ===== Generics (البرمجة العامة) =====
def first[T](items: list[T]) -> T:
    return items[0]

class Box[T]:
    def __init__(self, content: T) -> None:
        self.content = content

# ===== Protocol (duck typing مُهيكَل) =====
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> str: ...

# ===== mypy =====
# التثبيت:
#   uv add --dev mypy
# التشغيل:
#   uv run mypy file.py
#   uv run mypy . --strict
# الإعداد في pyproject.toml:
#   [tool.mypy]
#   strict = true

# ────────────────────────────────────────────────────────────────────────
# [090] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, percentage: float, customer: Customer) -> float:
    ...

# ────────────────────────────────────────────────────────────────────────
# [091] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
> from typing import List
> numbers: List[int] = [1, 2, 3]
> ```
> منذ Python 3.9، أصبح بالإمكان استخدام `list` مباشرة بحرف صغير دون استيراد أي شيء. وبما أننا نستهدف Python 3.14 في هذا الكتاب، سنعتمد دائمًا الصيغة الحديثة (`list[int]` وليس `List[int]`).

### الصناديق (Tuples): `tuple[T1, T2, ...]`

تتميّز الصناديق (Tuples) بأن كل عنصر فيها قد يكون من نوع مختلف، وعددها غالبًا ثابت:

# ────────────────────────────────────────────────────────────────────────
# [092] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
لاحظ أننا عرّفنا `Direction` كاسم مستعار (Type Alias) للنوع المركّب، لتجنّب تكرار كتابة `Literal["up", "down", "left", "right"]` في كل مكان.

> 💡 **نصيحة الخبير:** يجعل تعريف أسماء مستعارة للأنواع المعقّدة (Type Aliases) كودك أنظف بكثير:
> ```python
> UserId = int
> UserName = str
> UserDatabase = dict[UserId, UserName]
>
> def get_user(db: UserDatabase, uid: UserId) -> UserName | None:
>     return db.get(uid)
> ```
> ومنذ Python 3.12، توجد صيغة مخصصة وأوضح لهذا الغرض باستخدام الكلمة المفتاحية `type`:
> ```python
> type UserDatabase = dict[int, str]
> ```

---

## 15.9 القواميس المُهيكَلة: TypedDict

تذكّر القواميس من الفصل التاسع: مرنة جدًا، لكن مرونتها قد تنطوي على خطر. فماذا لو أردت قاموسًا يمثّل «بطاقة موظف» بحقول محددة ومعروفة مسبقًا (`name`, `age`, `salary`)، دون اللجوء إلى فئة (Class) كاملة لهذا الغرض البسيط؟

# ────────────────────────────────────────────────────────────────────────
# [093] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
هذه دالة وهمية خاصة بـ mypy فقط (لا تُستخدم فعليًا في الكود النهائي)، تُخبرك بالضبط عن النوع الذي استنتجته الأداة — مفيدة جدًا عند تصحيح مشاكل معقّدة.

### جدول مرجعي شامل: كل الأنواع التي تعلمناها

| الصيغة | المعنى | مثال |
|---|---|---|
| `int`, `float`, `str`, `bool` | أنواع أساسية | `age: int = 25` |
| `None` | لا شيء | `-> None` |
| `list[T]` | قائمة من نوع T | `list[int]` |
| `tuple[T1, T2]` | صندوق بأنواع محددة | `tuple[str, int]` |
| `tuple[T, ...]` | صندوق بعدد متغير من نفس النوع | `tuple[int, ...]` |
| `dict[K, V]` | قاموس | `dict[str, int]` |
| `set[T]` | مجموعة | `set[str]` |
| `T \| None` | اختياري (قد يكون None) | — |
| `T1 \| T2` | إمّا هذا أو ذاك | — |
| `Any` | أي نوع (بدون حماية) | `Any` |
| `Callable[[Args], Ret]` | دالة | `Callable[[int], str]` |
| `Literal[...]` | قيم محددة فقط | `Literal["a", "b"]` |
| `TypedDict` | قاموس بحقول معروفة | `class P(TypedDict): x: int` |
| `TypeVar` / `[T]` | نوع عام (Generic) | `def f[T](x: T) -> T` |
| `Protocol` | بروتوكول هيكلي | `class Drawable(Protocol): ...` |

---

## 15.14 مشروع تطبيقي: نظام إدارة مهام مُنمَّط بالكامل

لنطبّق كل ما تعلمناه في مشروع متكامل: نظام بسيط لإدارة المهام (Task Manager)، مع Type Hints كاملة ودقيقة، يجتاز فحص mypy في الوضع الصارم (`strict = true`) دون أي خطأ.

### المتطلبات

- كل مهمة لها: عنوان، أولوية (محدودة بقيم معينة فقط)، وحالة (معلّقة/قيد التنفيذ/منجزة).
- نظام يسمح بإضافة، البحث، تصفية، وتحديث حالة المهام.
- دعم Generic لأي نوع من «المستودعات» (Repositories) القابلة لإعادة الاستخدام.
- استخدام Protocol لتعريف «أي شيء قابل للعرض» (Displayable).

# ────────────────────────────────────────────────────────────────────────
# [094] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
> 💡 **نصيحة الخبير:** لاحظ استخدام `from __future__ import annotations` في بداية الملف. هذا السطر (مفيد خصوصًا في إصدارات أقدم من Python) يجعل كل Type Hints تُعامَل كنصوص (Strings) أثناء التشغيل، فلا تُقيَّم فعليًا، مما يُسرّع بدء تشغيل البرنامج قليلًا ويحل مشاكل «التوقيعات الدائرية» (عندما تحتاج فئة أن تُشير لنفسها أو لفئة لاحقة في الكود). في Python 3.14 أصبح هذا السلوك قريبًا من الافتراضي في بعض الحالات، لكن إضافته صراحة لا تزال ممارسة جيدة ومتوافقة مع إصدارات أخرى.

---

## 15.15 تمارين

### 🟢 مستوى مبتدئ

**تمرين 1:** أضف Type Hints كاملة لهذه الدالة (المعاملات وقيمة الإرجاع):

# ────────────────────────────────────────────────────────────────────────
# [095] dist/markdown/chapter-15.md
# ────────────────────────────────────────────────────────────────────────
</details>

---

## 15.16 اختبر فهمك (Quiz)

<details>
<summary>❓ السؤال 1: هل تمنع Python تشغيل الكود إن كانت الأنواع غير متطابقة مع Type Hints؟</summary>

لا. تتجاهل Python Type Hints تمامًا وقت التشغيل. هي «تلميحات» فقط موجَّهة للقارئ البشري ولأدوات خارجية مثل mypy والمحررات. يجب تشغيل أداة مثل mypy بشكل منفصل لاكتشاف هذه الأخطاء قبل التشغيل.

</details>

<details>
<summary>❓ السؤال 2: ما الفرق بين `list[int]` و `List[int]`؟</summary>

لا فرق في المعنى. `List[int]` هي الصيغة القديمة التي تتطلب `from typing import List`، بينما `list[int]` هي الصيغة الحديثة المدمجة في اللغة منذ Python 3.9، ولا تحتاج استيرادًا. نستخدم دائمًا `list[int]` في هذا الكتاب.

</details>

<details>
<summary>❓ السؤال 3: متى يجب استخدام `Any` بدل تحديد نوع دقيق؟</summary>

نادرًا جدًا، فقط عند التعامل مع بيانات غير معروفة الشكل فعليًا (كنتيجة لتحليل JSON خارجي)، أو أثناء ترحيل كود قديم تدريجيًا. يجب تجنّبها قدر الإمكان لأنها تُعطّل كل حماية نظام الأنواع على القيمة المعنية.

</details>

<details>
<summary>❓ السؤال 4: ما الفرق بين `Protocol` و `ABC` (Abstract Base Class)؟</summary>

تتطلب `ABC` وراثة صريحة (Nominal Typing) — يجب أن تُعلن الفئة أنها تَرث من الفئة المجردة. أما `Protocol` فلا يتطلب أي وراثة؛ يكفي أن تمتلك الفئة الدوال المطلوبة بنفس التوقيع (Structural Typing) — «إن كان شكلك مطابقًا، فأنت تحقق البروتوكول تلقائيًا»، بصرف النظر عن هرمية الوراثة.

</details>

<details>
<summary>❓ السؤال 5: ما وظيفة `TypeVar` أو الصيغة `[T]` في تعريف دالة؟</summary>

تسمح بكتابة دوال أو فئات عامة (Generic) تعمل مع أي نوع، مع الحفاظ على «الترابط» بين نوع المدخل ونوع المخرج. مثلًا، تضمن `def first[T](items: list[T]) -> T` أن ناتج الدالة هو نفس نوع عناصر القائمة المُمرَّرة، بصرف النظر عن النوع الفعلي.

</details>

<details>
<summary>❓ السؤال 6: لماذا نفضّل `str | None` على استخدام قيمة افتراضية وهمية مثل `""` للدلالة على «لا توجد قيمة»؟</summary>

لأن `""` نص صالح فعليًا وقد يكون قيمة حقيقية مقصودة (نص فارغ)، بينما يُعبّر `None` صراحة وبدون لبس عن «غياب القيمة تمامًا». استخدام `str | None` يجعل القصد واضحًا، ويجبرك (عبر mypy) على التحقق من `None` قبل استخدام القيمة، مما يمنع أخطاء شائعة.

</details>

---

## 15.17 ملخص الفصل (Cheat Sheet)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["mypy"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q mypy

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price, percentage, customer):
    ...

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
age: int = 25
name: str = "محمد"

def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# متغيرات
age: int = 25
price: float = 19.99
name: str = "سارة"
is_active: bool = True

# دوال
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"مرحبًا، {name}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")  # لا يوجد أي خطأ من Python!
print(result)  # "مرحبًا!"

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# قائمة أعداد صحيحة
numbers: list[int] = [1, 2, 3, 4, 5]

# قائمة نصوص
names: list[str] = ["أحمد", "سارة", "خالد"]

# قائمة متداخلة (قائمة من قوائم أعداد)
matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# صندوق يحتوي اسمًا (نص) وعمرًا (عدد صحيح)
person: tuple[str, int] = ("ليلى", 28)

# صندوق إحداثيات (x, y, z) - ثلاثة أعداد عشرية
point: tuple[float, float, float] = (1.5, 2.0, 3.5)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# صندوق بعدد غير محدد من الأعداد الصحيحة
scores: tuple[int, ...] = (90, 85, 100, 70, 60)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# قاموس: المفتاح نص، القيمة عدد صحيح
ages: dict[str, int] = {"أحمد": 30, "سارة": 25}

# قاموس أكثر تعقيدًا: المفتاح نص، القيمة قائمة نصوص
courses_by_student: dict[str, list[str]] = {
    "أحمد": ["رياضيات", "فيزياء"],
    "سارة": ["كيمياء"],
}

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
unique_ids: set[int] = {101, 102, 103}
tags: set[str] = {"python", "برمجة", "تعليم"}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def summarize_students(
    grades: dict[str, list[int]]
) -> dict[str, float]:
    """
    يستقبل قاموسًا بأسماء الطلاب ودرجاتهم،
    ويُرجع قاموسًا بأسماء الطلاب ومتوسط درجاتهم.
    """
    averages: dict[str, float] = {}
    for student, scores in grades.items():
        averages[student] = sum(scores) / len(scores)
    return averages

student_grades: dict[str, list[int]] = {
    "محمد": [85, 90, 78],
    "فاطمة": [92, 88, 95],
}

result: dict[str, float] = summarize_students(student_grades)
print(result)
# {'محمد': 84.33333333333333, 'فاطمة': 91.66666666666667}

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def find_user(user_id: int, users: dict[int, str]) -> str:
    return users.get(user_id)  # قد تُرجع None إن لم يوجد المستخدم!

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from typing import Optional

def find_user_old_style(user_id: int, users: dict[int, str]) -> Optional[str]:
    return users.get(user_id)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

result: str | None = find_user(5, {1: "أحمد", 2: "سارة"})

if result is not None:
    print(result.upper())  # آمن تمامًا، لأننا تحققنا أولًا
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# معامل اختياري (له قيمة افتراضية) - لا علاقة له بـ Optional type
def greet(name: str = "ضيف") -> str:
    return f"مرحبًا، {name}"

# Optional type - القيمة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def greet(name: str | None = None) -> str:
    if name is None:
        name = "ضيف"
    return f"مرحبًا، {name}"

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def format_id(value: int | str) -> str:
    """يحوّل معرّفًا (رقمًا أو نصًا) إلى نص منسّق."""
    return f"ID-{value}"

print(format_id(42))      # "ID-42"
print(format_id("A001"))  # "ID-A001"

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
from typing import Union

def format_id(value: Union[int, str]) -> str:
    return f"ID-{value}"

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def process(value: int | str) -> None:
    if isinstance(value, int):
        print(f"رقم مضاعف: {value * 2}")
    else:
        print(f"نص بأحرف كبيرة: {value.upper()}")

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
from typing import Any

def process_anything(data: Any) -> Any:
    return data

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
def func_any(x: Any) -> None:
    x.anything_at_all()  # لا تحذير من mypy حتى لو الدالة غير موجودة!

def func_object(x: object) -> None:
    x.anything_at_all()  # mypy يُصدر خطأً! object لا يملك هذه الدالة

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
from typing import Callable

def apply_operation(
    numbers: list[int],
    operation: Callable[[int], int]
) -> list[int]:
    """يطبّق 'operation' على كل عنصر في القائمة."""
    return [operation(n) for n in numbers]

def double(x: int) -> int:
    return x * 2

def square(x: int) -> int:
    return x * x

print(apply_operation([1, 2, 3], double))  # [2, 4, 6]
print(apply_operation([1, 2, 3], square))  # [1, 4, 9]

# يعمل أيضًا مع lambda
print(apply_operation([1, 2, 3], lambda x: x + 10))  # [11, 12, 13]

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# دالة بدون معاملات تُرجع نصًا
greeting_provider: Callable[[], str]

# دالة تستقبل int و str وتُرجع bool
validator: Callable[[int, str], bool]

# دالة بأي عدد ونوع من المعاملات (تجنّب هذا إن أمكن، لكنه موجود)
from typing import Callable
any_callable: Callable[..., int]  # ... تعني "أي معاملات"

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from typing import Callable
import time
import functools

def measure_time(func: Callable[..., float]) -> Callable[..., float]:
    @functools.wraps(func)
    def wrapper(*args: int, **kwargs: int) -> float:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"استغرقت {func.__name__}: {elapsed:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> float:
    return sum(i ** 2 for i in range(n)) / n

slow_calculation(1_000_000)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def get_first_element(items: list) -> Any:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
from typing import TypeVar

T = TypeVar("T")

def get_first_element(items: list[T]) -> T:
    return items[0]

numbers: list[int] = [10, 20, 30]
first_number = get_first_element(numbers)  # mypy يستنتج: int

words: list[str] = ["كتاب", "قلم"]
first_word = get_first_element(words)  # mypy يستنتج: str

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def get_first_element[T](items: list[T]) -> T:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
class Box[T]:
    """صندوق يحتفظ بعنصر واحد من أي نوع."""

    def __init__(self, content: T) -> None:
        self.content = content

    def get(self) -> T:
        return self.content

    def __repr__(self) -> str:
        return f"Box({self.content!r})"

int_box = Box(42)           # mypy يستنتج: Box[int]
str_box = Box("مرحبًا")      # mypy يستنتج: Box[str]

print(int_box.get())  # 42
print(str_box.get())  # مرحبًا

value: int = int_box.get()  # صحيح، mypy يعرف أن الناتج int

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
class Repository[T]:
    """مخزن بسيط يدير مجموعة من العناصر من نوع T."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        return self._items.copy()

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

from typing import Callable
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product_repo = Repository[Product]()
product_repo.add(Product("لابتوب", 1200.0))
product_repo.add(Product("هاتف", 800.0))

cheap_product = product_repo.find(lambda p: p.price < 1000)
print(cheap_product)  # Product(name='هاتف', price=800.0)

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "رسم دائرة ⭕"

class Square:
    def draw(self) -> str:
        return "رسم مربع ⬜"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # صحيح! Circle تملك draw()
render(Square())  # صحيح! Square تملك draw() أيضًا

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
user = find_user(5, {1: "أحمد"})
if user is not None:
    print(user.upper())
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
# ✅ جيد: التركيز على توقيع الدالة (المعاملات والإرجاع)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)  # mypy يستنتج أن subtotal من نوع float
    return subtotal * (1 + tax_rate)

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
"""
نظام إدارة المهام - مثال تطبيقي شامل على Type Hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, Callable

# ---------------------------------------------------------
# 1. تعريف الأنواع الأساسية باستخدام Literal و Type Alias
# ---------------------------------------------------------
type Priority = Literal["منخفضة", "متوسطة", "عالية"]
type Status = Literal["معلّقة", "قيد التنفيذ", "منجزة"]

# ---------------------------------------------------------
# 2. Protocol لأي شيء "قابل للعرض"
# ---------------------------------------------------------
class Displayable(Protocol):
    def display(self) -> str:
        ...

# ---------------------------------------------------------
# 3. فئة المهمة (Task) باستخدام dataclass
# ---------------------------------------------------------
@dataclass
class Task:
    title: str
    priority: Priority = "متوسطة"
    status: Status = "معلّقة"
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def display(self) -> str:
        due_info = f" (موعد التسليم: {self.due_date})" if self.due_date else ""
        tags_info = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"📋 {self.title} - {self.priority} - {self.status}{due_info}{tags_info}"

    def mark_as(self, new_status: Status) -> None:
        self.status = new_status

# ---------------------------------------------------------
# 4. مستودع عام (Generic Repository) لإدارة أي نوع من العناصر
# ---------------------------------------------------------
class Repository[T]:
    """مستودع عام يدير مجموعة من العناصر من أي نوع."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def remove(self, item: T) -> None:
        self._items.remove(item)

    def all(self) -> list[T]:
        return self._items.copy()

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        return [item for item in self._items if predicate(item)]

    def find_first(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(self._items)

# ---------------------------------------------------------
# 5. مدير المهام (TaskManager) - يستخدم Repository
# ---------------------------------------------------------
class TaskManager:
    def __init__(self) -> None:
        self._repo: Repository[Task] = Repository()

    def add_task(
        self,
        title: str,
        priority: Priority = "متوسطة",
        due_date: date | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task = Task(
            title=title,
            priority=priority,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        self._repo.add(task)
        return task

    def get_by_status(self, status: Status) -> list[Task]:
        return self._repo.filter(lambda t: t.status == status)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        return self._repo.filter(lambda t: t.priority == priority)

    def find_task_by_title(self, title: str) -> Task | None:
        return self._repo.find_first(lambda t: t.title == title)

    def complete_task(self, title: str) -> bool:
        task = self.find_task_by_title(title)
        if task is None:
            return False
        task.mark_as("منجزة")
        return True

    def summary(self) -> dict[Status, int]:
        statuses: list[Status] = ["معلّقة", "قيد التنفيذ", "منجزة"]
        return {
            status: len(self.get_by_status(status))
            for status in statuses
        }

    def display_all(self) -> None:
        if self._repo.count() == 0:
            print("لا توجد مهام بعد.")
            return
        for task in self._repo.all():
            print(task.display())

# ---------------------------------------------------------
# 6. دالة عامة تعمل مع أي كائن Displayable (Protocol)
# ---------------------------------------------------------
def print_report(items: list[Displayable], title: str = "تقرير") -> None:
    print(f"=== {title} ===")
    for item in items:
        print(item.display())
    print(f"عدد العناصر: {len(items)}")

# ---------------------------------------------------------
# 7. برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = TaskManager()

    manager.add_task(
        "كتابة تقرير شهري",
        priority="عالية",
        due_date=date(2025, 6, 30),
        tags=["عمل", "عاجل"],
    )
    manager.add_task("قراءة كتاب Python", priority="منخفضة")
    manager.add_task(
        "مراجعة الكود مع الفريق",
        priority="متوسطة",
        tags=["فريق"],
    )

    print("كل المهام:")
    manager.display_all()
    print()

    manager.complete_task("قراءة كتاب Python")

    print("المهام العالية الأولوية:")
    for task in manager.get_by_priority("عالية"):
        print(task.display())
    print()

    print("ملخص الحالات:")
    for status, count in manager.summary().items():
        print(f"  {status}: {count}")
    print()

    high_priority_tasks: list[Displayable] = list(manager.get_by_priority("عالية"))
    print_report(high_priority_tasks, title="تقرير المهام العاجلة")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def calculate_area(width, height):
    return width * height

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def calculate_area(width: float, height: float) -> float:
    return width * height

# تمرين 2
from typing import Literal

Grade = Literal["ممتاز", "جيد", "راسب"]

def get_grade(score: int) -> Grade:
    if score >= 90:
        return "ممتاز"
    elif score >= 60:
        return "جيد"
    else:
        return "راسب"

# تمرين 3
# لا فرق في المعنى: كلاهما يعني "قد يكون int أو None".
# Optional[int] هي الصيغة القديمة (تحتاج: from typing import Optional).
# int | None هي الصيغة الحديثة الموصى بها في Python 3.10+ (لا تحتاج استيراد).
# في كتابنا نستخدم دائمًا: int | None

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def find_product(product_id: int, catalog: dict[int, str]) -> str:
        return catalog.get(product_id)

    def get_discount(price: float, percent: int) -> float:
        return price - (price * percent)

    products = {1: "لابتوب", 2: "هاتف"}
    name = find_product(3, products)
    print(name.upper())
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # تمرين 4 - الكود المصحح
    def find_product(product_id: int, catalog: dict[int, str]) -> str | None:
        return catalog.get(product_id)  # قد تكون None فعلاً

    def get_discount(price: float, percent: int) -> float:
        return price - (price * (percent / 100))  # كانت تُطرح قيمة كبيرة جدًا خطأً منطقيًا

    products: dict[int, str] = {1: "لابتوب", 2: "هاتف"}
    name = find_product(3, products)

    if name is not None:
        print(name.upper())
    else:
        print("المنتج غير موجود")

    # تمرين 5
    class Stack[T]:
        def __init__(self) -> None:
            self._items: list[T] = []

        def push(self, item: T) -> None:
            self._items.append(item)

        def pop(self) -> T | None:
            if self.is_empty():
                return None
            return self._items.pop()

        def peek(self) -> T | None:
            if self.is_empty():
                return None
            return self._items[-1]

        def is_empty(self) -> bool:
            return len(self._items) == 0

    stack: Stack[int] = Stack()
    stack.push(1)
    stack.push(2)
    print(stack.pop())   # 2
    print(stack.peek())  # 1

    # تمرين 6
    from typing import Protocol

    class Comparable(Protocol):
        def __lt__(self, other: object) -> bool:
            ...

    def find_max[T: Comparable](items: list[T]) -> T:
        if not items:
            raise ValueError("القائمة فارغة")
        current_max = items[0]
        for item in items[1:]:
            if current_max < item:
                current_max = item
        return current_max

    print(find_max([3, 7, 2, 9, 4]))        # 9
    print(find_max(["بطة", "قطة", "أسد"]))  # قطة (حسب الترتيب الأبجدي)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
# ===== الأنواع الأساسية =====
age: int = 25
price: float = 19.99
name: str = "سارة"
active: bool = True

# ===== الهياكل المركّبة =====
numbers: list[int] = [1, 2, 3]
point: tuple[float, float] = (1.0, 2.0)
scores: tuple[int, ...] = (1, 2, 3, 4)
ages: dict[str, int] = {"أحمد": 30}
tags: set[str] = {"a", "b"}

# ===== القيم الاختيارية والمركّبة =====
result: str | None = None          # اختياري (الصيغة الحديثة)
value: int | str = 5               # إمّا هذا أو ذاك

# ===== Any (استخدمها بحذر!) =====
from typing import Any
data: Any = {"whatever": "shape"}

# ===== Callable (الدوال كقيم) =====
from typing import Callable
operation: Callable[[int, int], int]

# ===== Literal (قيم محددة فقط) =====
from typing import Literal
Status = Literal["pending", "done"]

# ===== TypedDict (قاموس مُهيكَل) =====
from typing import TypedDict
class User(TypedDict):
    name: str
    age: int

# ===== Generics (البرمجة العامة) =====
def first[T](items: list[T]) -> T:
    return items[0]

class Box[T]:
    def __init__(self, content: T) -> None:
        self.content = content

# ===== Protocol (duck typing مُهيكَل) =====
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> str: ...

# ===== mypy =====
# التثبيت:
#   uv add --dev mypy
# التشغيل:
#   uv run mypy file.py
#   uv run mypy . --strict
# الإعداد في pyproject.toml:
#   [tool.mypy]
#   strict = true

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-001-calculate_discount.py
"""

def calculate_discount(price, percentage, customer):
    ...

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-002-calculate_discount.py
"""

def calculate_discount(price: float, percentage: float, customer: Customer) -> float:
    ...

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.1: لماذا يستحق هذا الفصل اهتمامك الكامل؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-003-greet.py
"""

age: int = 25
name: str = "محمد"

def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-004-add.py
"""

# متغيرات
age: int = 25
price: float = 19.99
name: str = "سارة"
is_active: bool = True

# دوال
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"مرحبًا، {name}")

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.2: مراجعة سريعة: أين وصلنا حتى الآن؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-005-add.py
"""

def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")  # لا يوجد أي خطأ من Python!
print(result)  # "مرحبًا!"

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-006-ex15_3.py
"""

# قائمة أعداد صحيحة
numbers: list[int] = [1, 2, 3, 4, 5]

# قائمة نصوص
names: list[str] = ["أحمد", "سارة", "خالد"]

# قائمة متداخلة (قائمة من قوائم أعداد)
matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-007-ex15_3.py
"""

# صندوق يحتوي اسمًا (نص) وعمرًا (عدد صحيح)
person: tuple[str, int] = ("ليلى", 28)

# صندوق إحداثيات (x, y, z) - ثلاثة أعداد عشرية
point: tuple[float, float, float] = (1.5, 2.0, 3.5)

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-008-ex15_3.py
"""

# صندوق بعدد غير محدد من الأعداد الصحيحة
scores: tuple[int, ...] = (90, 85, 100, 70, 60)

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-009-ex15_3.py
"""

# قاموس: المفتاح نص، القيمة عدد صحيح
ages: dict[str, int] = {"أحمد": 30, "سارة": 25}

# قاموس أكثر تعقيدًا: المفتاح نص، القيمة قائمة نصوص
courses_by_student: dict[str, list[str]] = {
    "أحمد": ["رياضيات", "فيزياء"],
    "سارة": ["كيمياء"],
}

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-010-ex15_3.py
"""

unique_ids: set[int] = {101, 102, 103}
tags: set[str] = {"python", "برمجة", "تعليم"}

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.3: الأنواع المركّبة (Generic Types)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-011-summarize_students.py
"""

def summarize_students(
    grades: dict[str, list[int]]
) -> dict[str, float]:
    """
    يستقبل قاموسًا بأسماء الطلاب ودرجاتهم،
    ويُرجع قاموسًا بأسماء الطلاب ومتوسط درجاتهم.
    """
    averages: dict[str, float] = {}
    for student, scores in grades.items():
        averages[student] = sum(scores) / len(scores)
    return averages

student_grades: dict[str, list[int]] = {
    "محمد": [85, 90, 78],
    "فاطمة": [92, 88, 95],
}

result: dict[str, float] = summarize_students(student_grades)
print(result)
# {'محمد': 84.33333333333333, 'فاطمة': 91.66666666666667}

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-012-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str:
    return users.get(user_id)  # قد تُرجع None إن لم يوجد المستخدم!

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-013-find_user_old_style.py
"""

from typing import Optional

def find_user_old_style(user_id: int, users: dict[int, str]) -> Optional[str]:
    return users.get(user_id)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-014-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

result: str | None = find_user(5, {1: "أحمد", 2: "سارة"})

if result is not None:
    print(result.upper())  # آمن تمامًا، لأننا تحققنا أولًا
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-015-greet.py
"""

# معامل اختياري (له قيمة افتراضية) - لا علاقة له بـ Optional type
def greet(name: str = "ضيف") -> str:
    return f"مرحبًا، {name}"

# Optional type - القيمة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-016-greet.py
"""

def greet(name: str | None = None) -> str:
    if name is None:
        name = "ضيف"
    return f"مرحبًا، {name}"

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-017-format_id.py
"""

def format_id(value: int | str) -> str:
    """يحوّل معرّفًا (رقمًا أو نصًا) إلى نص منسّق."""
    return f"ID-{value}"

print(format_id(42))      # "ID-42"
print(format_id("A001"))  # "ID-A001"

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-018-format_id.py
"""

from typing import Union

def format_id(value: Union[int, str]) -> str:
    return f"ID-{value}"

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.5: عندما يكون هناك أكثر من نوع ممكن: Union

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-019-process.py
"""

def process(value: int | str) -> None:
    if isinstance(value, int):
        print(f"رقم مضاعف: {value * 2}")
    else:
        print(f"نص بأحرف كبيرة: {value.upper()}")

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-020-process_anything.py
"""

from typing import Any

def process_anything(data: Any) -> Any:
    return data

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.6: النوع الهارب: Any

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-021-func_any.py
"""

def func_any(x: Any) -> None:
    x.anything_at_all()  # لا تحذير من mypy حتى لو الدالة غير موجودة!

def func_object(x: object) -> None:
    x.anything_at_all()  # mypy يُصدر خطأً! object لا يملك هذه الدالة

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-022-apply_operation.py
"""

from typing import Callable

def apply_operation(
    numbers: list[int],
    operation: Callable[[int], int]
) -> list[int]:
    """يطبّق 'operation' على كل عنصر في القائمة."""
    return [operation(n) for n in numbers]

def double(x: int) -> int:
    return x * 2

def square(x: int) -> int:
    return x * x

print(apply_operation([1, 2, 3], double))  # [2, 4, 6]
print(apply_operation([1, 2, 3], square))  # [1, 4, 9]

# يعمل أيضًا مع lambda
print(apply_operation([1, 2, 3], lambda x: x + 10))  # [11, 12, 13]

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-023-ex15_7.py
"""

# دالة بدون معاملات تُرجع نصًا
greeting_provider: Callable[[], str]

# دالة تستقبل int و str وتُرجع bool
validator: Callable[[int, str], bool]

# دالة بأي عدد ونوع من المعاملات (تجنّب هذا إن أمكن، لكنه موجود)
from typing import Callable
any_callable: Callable[..., int]  # ... تعني "أي معاملات"

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-024-measure_time.py
"""

from typing import Callable
import time
import functools

def measure_time(func: Callable[..., float]) -> Callable[..., float]:
    @functools.wraps(func)
    def wrapper(*args: int, **kwargs: int) -> float:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"استغرقت {func.__name__}: {elapsed:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> float:
    return sum(i ** 2 for i in range(n)) / n

slow_calculation(1_000_000)

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-025-set_status.py
"""

from typing import Literal

def set_status(status: Literal["active", "inactive", "pending"]) -> None:
    print(f"الحالة الجديدة: {status}")

set_status("active")    # صحيح
set_status("pending")   # صحيح
set_status("deleted")   # ❌ mypy يرفض هذا! "deleted" ليست من القيم المسموحة

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-026-move.py
"""

Direction = Literal["up", "down", "left", "right"]

def move(direction: Direction, steps: int = 1) -> tuple[int, int]:
    deltas: dict[Direction, tuple[int, int]] = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    dx, dy = deltas[direction]
    return (dx * steps, dy * steps)

print(move("up", 3))  # (0, 3)

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.9: القواميس المُهيكَلة: TypedDict

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-027-print_employee.py
"""

from typing import TypedDict

class Employee(TypedDict):
    name: str
    age: int
    salary: float

def print_employee(emp: Employee) -> None:
    print(f"{emp['name']} - {emp['age']} سنة - راتب {emp['salary']} دولار")

employee1: Employee = {
    "name": "ياسر",
    "age": 35,
    "salary": 5000.0,
}

print_employee(employee1)

# استخدام خاطئ يكتشفه mypy:
bad_employee: Employee = {
    "name": "سلمى",
    "age": "ثلاثون",  # ❌ خطأ! يجب أن يكون int وليس str
    "salary": 4500.0,
}

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-028-get_first_element.py
"""

def get_first_element(items: list) -> Any:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-029-get_first_element.py
"""

from typing import TypeVar

T = TypeVar("T")

def get_first_element(items: list[T]) -> T:
    return items[0]

numbers: list[int] = [10, 20, 30]
first_number = get_first_element(numbers)  # mypy يستنتج: int

words: list[str] = ["كتاب", "قلم"]
first_word = get_first_element(words)  # mypy يستنتج: str

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-030-get_first_element.py
"""

def get_first_element[T](items: list[T]) -> T:
    return items[0]

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-031-__init__.py
"""

class Box[T]:
    """صندوق يحتفظ بعنصر واحد من أي نوع."""

    def __init__(self, content: T) -> None:
        self.content = content

    def get(self) -> T:
        return self.content

    def __repr__(self) -> str:
        return f"Box({self.content!r})"

int_box = Box(42)           # mypy يستنتج: Box[int]
str_box = Box("مرحبًا")      # mypy يستنتج: Box[str]

print(int_box.get())  # 42
print(str_box.get())  # مرحبًا

value: int = int_box.get()  # صحيح، mypy يعرف أن الناتج int

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.10: البرمجة العامة (Generics) باستخدام TypeVar

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-032-__init__.py
"""

class Repository[T]:
    """مخزن بسيط يدير مجموعة من العناصر من نوع T."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        return self._items.copy()

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

from typing import Callable
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product_repo = Repository[Product]()
product_repo.add(Product("لابتوب", 1200.0))
product_repo.add(Product("هاتف", 800.0))

cheap_product = product_repo.find(lambda p: p.price < 1000)
print(cheap_product)  # Product(name='هاتف', price=800.0)

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.11: البروتوكولات: Protocol (Duck Typing المُهيكَل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-033-draw.py
"""

from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "رسم دائرة ⭕"

class Square:
    def draw(self) -> str:
        return "رسم مربع ⬜"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # صحيح! Circle تملك draw()
render(Square())  # صحيح! Square تملك draw() أيضًا

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-034-add.py
"""

def add(a: int, b: int) -> int:
    return a + b

result = add("مرحبًا", "!")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-035-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

user = find_user(5, {1: "أحمد"})
print(user.upper())  # ❌ خطأ محتمل!

# ────────────────────────────────────────────────────────────────────────
# [076] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-036-user.py
"""

user = find_user(5, {1: "أحمد"})
if user is not None:
    print(user.upper())
else:
    print("المستخدم غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [077] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-76
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-037-greet.py
"""

def greet(name: str, greeting: str = "مرحبًا") -> str:
    return f"{greeting}، {name}!"

print(greet())  # ❌ نسينا معاملًا إلزاميًا

# ────────────────────────────────────────────────────────────────────────
# [078] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-77
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-038-get_age.py
"""

def get_age() -> int:
    return "عشرون"  # ❌ نص بدل عدد صحيح

# ────────────────────────────────────────────────────────────────────────
# [079] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-78
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-039-point.py
"""

from typing import TypedDict

class Point(TypedDict):
    x: int
    y: int

p: Point = {"x": 1}  # ❌ ناقص حقل y

# ────────────────────────────────────────────────────────────────────────
# [080] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-79
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-040-result.py
"""

import some_untyped_library  # type: ignore[import-untyped]

result = some_untyped_library.do_something()

# ────────────────────────────────────────────────────────────────────────
# [081] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-80
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-041-calculate_total.py
"""

# ✅ جيد: التركيز على توقيع الدالة (المعاملات والإرجاع)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)  # mypy يستنتج أن subtotal من نوع float
    return subtotal * (1 + tax_rate)

# ────────────────────────────────────────────────────────────────────────
# [082] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-81
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-042-find.py
"""

# ❌ غامض: هل يمكن أن يكون None؟ لا نعرف
def find(key: str) -> str:
    ...

# ✅ واضح: نعرف بالضبط أن النتيجة قد تكون None
def find(key: str) -> str | None:
    ...

# ────────────────────────────────────────────────────────────────────────
# [083] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-82
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.13: أفضل الممارسات العامة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-043-ex15_13.py
"""

x = [1, 2, 3]
reveal_type(x)  # أداة تشخيصية خاصة بـ mypy فقط

# ────────────────────────────────────────────────────────────────────────
# [084] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-83
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.14: مشروع تطبيقي: نظام إدارة مهام مُنمَّط بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-044-display.py
"""

"""
نظام إدارة المهام - مثال تطبيقي شامل على Type Hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, Callable

# ---------------------------------------------------------
# 1. تعريف الأنواع الأساسية باستخدام Literal و Type Alias
# ---------------------------------------------------------
type Priority = Literal["منخفضة", "متوسطة", "عالية"]
type Status = Literal["معلّقة", "قيد التنفيذ", "منجزة"]

# ---------------------------------------------------------
# 2. Protocol لأي شيء "قابل للعرض"
# ---------------------------------------------------------
class Displayable(Protocol):
    def display(self) -> str:
        ...

# ---------------------------------------------------------
# 3. فئة المهمة (Task) باستخدام dataclass
# ---------------------------------------------------------
@dataclass
class Task:
    title: str
    priority: Priority = "متوسطة"
    status: Status = "معلّقة"
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def display(self) -> str:
        due_info = f" (موعد التسليم: {self.due_date})" if self.due_date else ""
        tags_info = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"📋 {self.title} - {self.priority} - {self.status}{due_info}{tags_info}"

    def mark_as(self, new_status: Status) -> None:
        self.status = new_status

# ---------------------------------------------------------
# 4. مستودع عام (Generic Repository) لإدارة أي نوع من العناصر
# ---------------------------------------------------------
class Repository[T]:
    """مستودع عام يدير مجموعة من العناصر من أي نوع."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def remove(self, item: T) -> None:
        self._items.remove(item)

    def all(self) -> list[T]:
        return self._items.copy()

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        return [item for item in self._items if predicate(item)]

    def find_first(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(self._items)

# ---------------------------------------------------------
# 5. مدير المهام (TaskManager) - يستخدم Repository
# ---------------------------------------------------------
class TaskManager:
    def __init__(self) -> None:
        self._repo: Repository[Task] = Repository()

    def add_task(
        self,
        title: str,
        priority: Priority = "متوسطة",
        due_date: date | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task = Task(
            title=title,
            priority=priority,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        self._repo.add(task)
        return task

    def get_by_status(self, status: Status) -> list[Task]:
        return self._repo.filter(lambda t: t.status == status)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        return self._repo.filter(lambda t: t.priority == priority)

    def find_task_by_title(self, title: str) -> Task | None:
        return self._repo.find_first(lambda t: t.title == title)

    def complete_task(self, title: str) -> bool:
        task = self.find_task_by_title(title)
        if task is None:
            return False
        task.mark_as("منجزة")
        return True

    def summary(self) -> dict[Status, int]:
        statuses: list[Status] = ["معلّقة", "قيد التنفيذ", "منجزة"]
        return {
            status: len(self.get_by_status(status))
            for status in statuses
        }

    def display_all(self) -> None:
        if self._repo.count() == 0:
            print("لا توجد مهام بعد.")
            return
        for task in self._repo.all():
            print(task.display())

# ---------------------------------------------------------
# 6. دالة عامة تعمل مع أي كائن Displayable (Protocol)
# ---------------------------------------------------------
def print_report(items: list[Displayable], title: str = "تقرير") -> None:
    print(f"=== {title} ===")
    for item in items:
        print(item.display())
    print(f"عدد العناصر: {len(items)}")

# ---------------------------------------------------------
# 7. برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = TaskManager()

    manager.add_task(
        "كتابة تقرير شهري",
        priority="عالية",
        due_date=date(2025, 6, 30),
        tags=["عمل", "عاجل"],
    )
    manager.add_task("قراءة كتاب Python", priority="منخفضة")
    manager.add_task(
        "مراجعة الكود مع الفريق",
        priority="متوسطة",
        tags=["فريق"],
    )

    print("كل المهام:")
    manager.display_all()
    print()

    manager.complete_task("قراءة كتاب Python")

    print("المهام العالية الأولوية:")
    for task in manager.get_by_priority("عالية"):
        print(task.display())
    print()

    print("ملخص الحالات:")
    for status, count in manager.summary().items():
        print(f"  {status}: {count}")
    print()

    high_priority_tasks: list[Displayable] = list(manager.get_by_priority("عالية"))
    print_report(high_priority_tasks, title="تقرير المهام العاجلة")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [085] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-84
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-045-calculate_area.py
"""

def calculate_area(width, height):
    return width * height

# ────────────────────────────────────────────────────────────────────────
# [086] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-85
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-046-calculate_area.py
"""

# تمرين 1
def calculate_area(width: float, height: float) -> float:
    return width * height

# تمرين 2
from typing import Literal

Grade = Literal["ممتاز", "جيد", "راسب"]

def get_grade(score: int) -> Grade:
    if score >= 90:
        return "ممتاز"
    elif score >= 60:
        return "جيد"
    else:
        return "راسب"

# تمرين 3
# لا فرق في المعنى: كلاهما يعني "قد يكون int أو None".
# Optional[int] هي الصيغة القديمة (تحتاج: from typing import Optional).
# int | None هي الصيغة الحديثة الموصى بها في Python 3.10+ (لا تحتاج استيراد).
# في كتابنا نستخدم دائمًا: int | None

# ────────────────────────────────────────────────────────────────────────
# [087] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-86
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-047-find_product.py
"""

def find_product(product_id: int, catalog: dict[int, str]) -> str:
    return catalog.get(product_id)

def get_discount(price: float, percent: int) -> float:
    return price - (price * percent)

products = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)
print(name.upper())

# ────────────────────────────────────────────────────────────────────────
# [088] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-87
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-048-find_product.py
"""

# تمرين 4 - الكود المصحح
def find_product(product_id: int, catalog: dict[int, str]) -> str | None:
    return catalog.get(product_id)  # قد تكون None فعلاً

def get_discount(price: float, percent: int) -> float:
    return price - (price * (percent / 100))  # كانت تُطرح قيمة كبيرة جدًا خطأً منطقيًا

products: dict[int, str] = {1: "لابتوب", 2: "هاتف"}
name = find_product(3, products)

if name is not None:
    print(name.upper())
else:
    print("المنتج غير موجود")

# تمرين 5
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T | None:
        if self.is_empty():
            return None
        return self._items.pop()

    def peek(self) -> T | None:
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())   # 2
print(stack.peek())  # 1

# تمرين 6
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other: object) -> bool:
        ...

def find_max[T: Comparable](items: list[T]) -> T:
    if not items:
        raise ValueError("القائمة فارغة")
    current_max = items[0]
    for item in items[1:]:
        if current_max < item:
            current_max = item
    return current_max

print(find_max([3, 7, 2, 9, 4]))        # 9
print(find_max(["بطة", "قطة", "أسد"]))  # قطة (حسب الترتيب الأبجدي)

# ────────────────────────────────────────────────────────────────────────
# [089] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-88
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-049-__init__.py
"""

from typing import TypedDict, Callable

class CacheStats(TypedDict):
    hits: int
    misses: int

class Cache[K, V]:
    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: K) -> V | None:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get_or_compute(self, key: K, compute_fn: Callable[[], V]) -> V:
        existing = self.get(key)
        if existing is not None:
            return existing
        new_value = compute_fn()
        self.set(key, new_value)
        return new_value

    def stats(self) -> CacheStats:
        return {"hits": self._hits, "misses": self._misses}

def expensive_computation(n: int) -> int:
    print(f"🔄 جارِ الحساب الباهظ لـ {n} ...")
    return n ** 2

cache: Cache[int, int] = Cache()

result1 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result1)  # يطبع "جارِ الحساب..." ثم 25

result2 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result2)  # لا يطبع "جارِ الحساب" لأنها مخزّنة! يطبع 25 مباشرة

print(cache.stats())  # {'hits': 1, 'misses': 1}

# ────────────────────────────────────────────────────────────────────────
# [090] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-89
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.17: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-050-first.py
"""

# ===== الأنواع الأساسية =====
age: int = 25
price: float = 19.99
name: str = "سارة"
active: bool = True

# ===== الهياكل المركّبة =====
numbers: list[int] = [1, 2, 3]
point: tuple[float, float] = (1.0, 2.0)
scores: tuple[int, ...] = (1, 2, 3, 4)
ages: dict[str, int] = {"أحمد": 30}
tags: set[str] = {"a", "b"}

# ===== القيم الاختيارية والمركّبة =====
result: str | None = None          # اختياري (الصيغة الحديثة)
value: int | str = 5               # إمّا هذا أو ذاك

# ===== Any (استخدمها بحذر!) =====
from typing import Any
data: Any = {"whatever": "shape"}

# ===== Callable (الدوال كقيم) =====
from typing import Callable
operation: Callable[[int, int], int]

# ===== Literal (قيم محددة فقط) =====
from typing import Literal
Status = Literal["pending", "done"]

# ===== TypedDict (قاموس مُهيكَل) =====
from typing import TypedDict
class User(TypedDict):
    name: str
    age: int

# ===== Generics (البرمجة العامة) =====
def first[T](items: list[T]) -> T:
    return items[0]

class Box[T]:
    def __init__(self, content: T) -> None:
        self.content = content

# ===== Protocol (duck typing مُهيكَل) =====
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> str: ...

# ===== mypy =====
# التثبيت:
#   uv add --dev mypy
# التشغيل:
#   uv run mypy file.py
#   uv run mypy . --strict
# الإعداد في pyproject.toml:
#   [tool.mypy]
#   strict = true

# ────────────────────────────────────────────────────────────────────────
# [091] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-90
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, percentage: float, customer: Customer) -> float:
    ...

# ────────────────────────────────────────────────────────────────────────
# [092] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-91
# ────────────────────────────────────────────────────────────────────────
> from typing import List
> numbers: List[int] = [1, 2, 3]
> ```
> منذ Python 3.9، أصبح بالإمكان استخدام `list` مباشرة بحرف صغير دون استيراد أي شيء. وبما أننا نستهدف Python 3.14 في هذا الكتاب، سنعتمد دائمًا الصيغة الحديثة (`list[int]` وليس `List[int]`).

### الصناديق (Tuples): `tuple[T1, T2, ...]`

تتميّز الصناديق (Tuples) بأن كل عنصر فيها قد يكون من نوع مختلف، وعددها غالبًا ثابت:

# ────────────────────────────────────────────────────────────────────────
# [093] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-92
# ────────────────────────────────────────────────────────────────────────
لاحظ أننا عرّفنا `Direction` كاسم مستعار (Type Alias) للنوع المركّب، لتجنّب تكرار كتابة `Literal["up", "down", "left", "right"]` في كل مكان.

> 💡 **نصيحة الخبير:** يجعل تعريف أسماء مستعارة للأنواع المعقّدة (Type Aliases) كودك أنظف بكثير:
> ```python
> UserId = int
> UserName = str
> UserDatabase = dict[UserId, UserName]
>
> def get_user(db: UserDatabase, uid: UserId) -> UserName | None:
>     return db.get(uid)
> ```
> ومنذ Python 3.12، توجد صيغة مخصصة وأوضح لهذا الغرض باستخدام الكلمة المفتاحية `type`:
> ```python
> type UserDatabase = dict[int, str]
> ```

---

## 15.9 القواميس المُهيكَلة: TypedDict

تذكّر القواميس من الفصل التاسع: مرنة جدًا، لكن مرونتها قد تنطوي على خطر. فماذا لو أردت قاموسًا يمثّل «بطاقة موظف» بحقول محددة ومعروفة مسبقًا (`name`, `age`, `salary`)، دون اللجوء إلى فئة (Class) كاملة لهذا الغرض البسيط؟

# ────────────────────────────────────────────────────────────────────────
# [094] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-93
# ────────────────────────────────────────────────────────────────────────
هذه دالة وهمية خاصة بـ mypy فقط (لا تُستخدم فعليًا في الكود النهائي)، تُخبرك بالضبط عن النوع الذي استنتجته الأداة — مفيدة جدًا عند تصحيح مشاكل معقّدة.

### جدول مرجعي شامل: كل الأنواع التي تعلمناها

| الصيغة | المعنى | مثال |
|---|---|---|
| `int`, `float`, `str`, `bool` | أنواع أساسية | `age: int = 25` |
| `None` | لا شيء | `-> None` |
| `list[T]` | قائمة من نوع T | `list[int]` |
| `tuple[T1, T2]` | صندوق بأنواع محددة | `tuple[str, int]` |
| `tuple[T, ...]` | صندوق بعدد متغير من نفس النوع | `tuple[int, ...]` |
| `dict[K, V]` | قاموس | `dict[str, int]` |
| `set[T]` | مجموعة | `set[str]` |
| `T \| None` | اختياري (قد يكون None) | — |
| `T1 \| T2` | إمّا هذا أو ذاك | — |
| `Any` | أي نوع (بدون حماية) | `Any` |
| `Callable[[Args], Ret]` | دالة | `Callable[[int], str]` |
| `Literal[...]` | قيم محددة فقط | `Literal["a", "b"]` |
| `TypedDict` | قاموس بحقول معروفة | `class P(TypedDict): x: int` |
| `TypeVar` / `[T]` | نوع عام (Generic) | `def f[T](x: T) -> T` |
| `Protocol` | بروتوكول هيكلي | `class Drawable(Protocol): ...` |

---

## 15.14 مشروع تطبيقي: نظام إدارة مهام مُنمَّط بالكامل

لنطبّق كل ما تعلمناه في مشروع متكامل: نظام بسيط لإدارة المهام (Task Manager)، مع Type Hints كاملة ودقيقة، يجتاز فحص mypy في الوضع الصارم (`strict = true`) دون أي خطأ.

### المتطلبات

- كل مهمة لها: عنوان، أولوية (محدودة بقيم معينة فقط)، وحالة (معلّقة/قيد التنفيذ/منجزة).
- نظام يسمح بإضافة، البحث، تصفية، وتحديث حالة المهام.
- دعم Generic لأي نوع من «المستودعات» (Repositories) القابلة لإعادة الاستخدام.
- استخدام Protocol لتعريف «أي شيء قابل للعرض» (Displayable).

# ────────────────────────────────────────────────────────────────────────
# [095] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-94
# ────────────────────────────────────────────────────────────────────────
> 💡 **نصيحة الخبير:** لاحظ استخدام `from __future__ import annotations` في بداية الملف. هذا السطر (مفيد خصوصًا في إصدارات أقدم من Python) يجعل كل Type Hints تُعامَل كنصوص (Strings) أثناء التشغيل، فلا تُقيَّم فعليًا، مما يُسرّع بدء تشغيل البرنامج قليلًا ويحل مشاكل «التوقيعات الدائرية» (عندما تحتاج فئة أن تُشير لنفسها أو لفئة لاحقة في الكود). في Python 3.14 أصبح هذا السلوك قريبًا من الافتراضي في بعض الحالات، لكن إضافته صراحة لا تزال ممارسة جيدة ومتوافقة مع إصدارات أخرى.

---

## 15.15 تمارين

### 🟢 مستوى مبتدئ

**تمرين 1:** أضف Type Hints كاملة لهذه الدالة (المعاملات وقيمة الإرجاع):

# ────────────────────────────────────────────────────────────────────────
# [096] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-15-chapter-15.ipynb#cell-95
# ────────────────────────────────────────────────────────────────────────
</details>

---

## 15.16 اختبر فهمك (Quiz)

<details>
<summary>❓ السؤال 1: هل تمنع Python تشغيل الكود إن كانت الأنواع غير متطابقة مع Type Hints؟</summary>

لا. تتجاهل Python Type Hints تمامًا وقت التشغيل. هي «تلميحات» فقط موجَّهة للقارئ البشري ولأدوات خارجية مثل mypy والمحررات. يجب تشغيل أداة مثل mypy بشكل منفصل لاكتشاف هذه الأخطاء قبل التشغيل.

</details>

<details>
<summary>❓ السؤال 2: ما الفرق بين `list[int]` و `List[int]`؟</summary>

لا فرق في المعنى. `List[int]` هي الصيغة القديمة التي تتطلب `from typing import List`، بينما `list[int]` هي الصيغة الحديثة المدمجة في اللغة منذ Python 3.9، ولا تحتاج استيرادًا. نستخدم دائمًا `list[int]` في هذا الكتاب.

</details>

<details>
<summary>❓ السؤال 3: متى يجب استخدام `Any` بدل تحديد نوع دقيق؟</summary>

نادرًا جدًا، فقط عند التعامل مع بيانات غير معروفة الشكل فعليًا (كنتيجة لتحليل JSON خارجي)، أو أثناء ترحيل كود قديم تدريجيًا. يجب تجنّبها قدر الإمكان لأنها تُعطّل كل حماية نظام الأنواع على القيمة المعنية.

</details>

<details>
<summary>❓ السؤال 4: ما الفرق بين `Protocol` و `ABC` (Abstract Base Class)؟</summary>

تتطلب `ABC` وراثة صريحة (Nominal Typing) — يجب أن تُعلن الفئة أنها تَرث من الفئة المجردة. أما `Protocol` فلا يتطلب أي وراثة؛ يكفي أن تمتلك الفئة الدوال المطلوبة بنفس التوقيع (Structural Typing) — «إن كان شكلك مطابقًا، فأنت تحقق البروتوكول تلقائيًا»، بصرف النظر عن هرمية الوراثة.

</details>

<details>
<summary>❓ السؤال 5: ما وظيفة `TypeVar` أو الصيغة `[T]` في تعريف دالة؟</summary>

تسمح بكتابة دوال أو فئات عامة (Generic) تعمل مع أي نوع، مع الحفاظ على «الترابط» بين نوع المدخل ونوع المخرج. مثلًا، تضمن `def first[T](items: list[T]) -> T` أن ناتج الدالة هو نفس نوع عناصر القائمة المُمرَّرة، بصرف النظر عن النوع الفعلي.

</details>

<details>
<summary>❓ السؤال 6: لماذا نفضّل `str | None` على استخدام قيمة افتراضية وهمية مثل `""` للدلالة على «لا توجد قيمة»؟</summary>

لأن `""` نص صالح فعليًا وقد يكون قيمة حقيقية مقصودة (نص فارغ)، بينما يُعبّر `None` صراحة وبدون لبس عن «غياب القيمة تمامًا». استخدام `str | None` يجعل القصد واضحًا، ويجبرك (عبر mypy) على التحقق من `None` قبل استخدام القيمة، مما يمنع أخطاء شائعة.

</details>

---

## 15.17 ملخص الفصل (Cheat Sheet)
