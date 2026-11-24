#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملحق ج: جدول مرجعي سريع (Cheat Sheet) شامل
Python 3.14 — From Zero to Pro — C

Book slug : python-3-14-from-zero-to-pro
Chapter   : app-c-appendix-c (lesson c)
Code files: 26 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/app-c-appendix-c.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملحق ج: جدول مرجعي سريع (Cheat Sheet) شامل
Python 3.14 — From Zero to Pro — C

Book slug : python-3-14-from-zero-to-pro
Chapter   : app-c-appendix-c (lesson c)
Code files: 25 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
# التعريف مع Type Hints
name: str = "محمد"
age: int = 30
price: float = 19.99
is_active: bool = True
data: None = None

# تحويل الأنواع
int("42")          # 42
float("3.14")      # 3.14
str(100)           # "100"
bool(0)            # False
list("abc")        # ['a', 'b', 'c']

# فحص النوع
type(x)             # النوع الفعلي
isinstance(x, int)  # فحص إن كان من نوع معيّن

# ────────────────────────────────────────────────────────────────────────
# [002] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
+  -  *  /       # عمليات أساسية (/ تُنتج float دائمًا)
//              # قسمة صحيحة (floor division)
# %               # باقي القسمة
**              # قوة (أس)

==  !=  <  >  <=  >=    # مقارنة
and  or  not             # منطقية

# ────────────────────────────────────────────────────────────────────────
# [003] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
name, age = "سارة", 25
f"{name} عمرها {age}"
f"{price:.2f}"           # تنسيق عشري بمنزلتين
f"{number:,}"            # فاصلة آلاف
f"{value:>10}"           # محاذاة يمين بعرض 10
f"{value=}"              # يطبع "value=10" (مفيد للتنقيح)

# ────────────────────────────────────────────────────────────────────────
# [004] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
s = "Hello World"

s.upper()           # "HELLO WORLD"
s.lower()           # "hello world"
s.strip()           # إزالة المسافات من الطرفين
s.replace("a", "b") # استبدال
s.split(" ")        # تقسيم لقائمة
"-".join(["a","b"]) # دمج قائمة لنص
s.startswith("He")  # True
s.endswith("ld")    # True
s.find("World")     # 6 (أو -1 إن لم يوجد)
len(s)              # طول النص
s[0]                # أول حرف
s[-1]               # آخر حرف
s[0:5]              # تقطيع (Slicing)
s[::-1]             # عكس النص

# ────────────────────────────────────────────────────────────────────────
# [005] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
lst = [1, 2, 3]

lst.append(4)        # إضافة في النهاية
lst.insert(0, 0)      # إضافة في موضع محدد
lst.remove(2)         # إزالة أول قيمة مطابقة
lst.pop()             # إزالة وإرجاع آخر عنصر
lst.pop(0)            # إزالة وإرجاع عنصر بموضع محدد
lst.sort()            # ترتيب في المكان
sorted(lst)           # ترتيب في نسخة جديدة
lst.reverse()         # عكس في المكان
len(lst)              # الطول
lst[1:3]              # تقطيع
2 in lst              # فحص الوجود

# List Comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [006] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
d = {"name": "علي", "age": 30}

d["name"]              # قراءة
d["city"] = "الرياض"   # إضافة/تعديل
d.get("key", default)  # قراءة آمنة بقيمة افتراضية
d.keys()                # المفاتيح
d.values()              # القيم
d.items()               # أزواج (مفتاح، قيمة)
del d["age"]            # حذف
"name" in d             # فحص وجود مفتاح

# Dict Comprehension
squares = {x: x**2 for x in range(5)}

# ────────────────────────────────────────────────────────────────────────
# [007] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
s = {1, 2, 3}

s.add(4)
s.remove(2)
s1 | s2    # اتحاد
s1 & s2    # تقاطع
s1 - s2    # فرق
s1 ^ s2    # فرق متماثل

# ────────────────────────────────────────────────────────────────────────
# [008] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
t = (1, 2, 3)
a, b, c = t          # فك التغليف (Unpacking)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
# شرطية
if condition:
    ...
elif other_condition:
    ...
else:
    ...

# التعبير الشرطي (Ternary)
result = "نعم" if condition else "لا"

# حلقة for
for item in iterable:
    ...
for i in range(10):          # 0 → 9
for i, item in enumerate(lst):  # مع الفهرس
for a, b in zip(list1, list2):  # تكرار متوازٍ

# حلقة while
while condition:
    ...

# كلمات تحكم
break       # إيقاف الحلقة
continue    # تخطي الدورة الحالية
pass        # عدم فعل شيء (حجز مكان)

# ────────────────────────────────────────────────────────────────────────
# [010] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "مرحبًا") -> str:
    """Docstring يشرح الدالة."""
    return f"{greeting}, {name}!"

# وسائط متغيرة
def total(*args: int) -> int:
    return sum(args)

def config(**kwargs: str) -> None:
    for key, value in kwargs.items():
        print(key, value)

# Lambda
square = lambda x: x ** 2

# Type Hints شائعة
def process(items: list[str]) -> dict[str, int]: ...
def maybe(value: int | None = None) -> bool: ...

# ────────────────────────────────────────────────────────────────────────
# [011] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except ValueError as e:
    print(f"خطأ: {e}")
except (TypeError, KeyError):
    print("أحد هذين الخطأين")
else:
    print("نُفِّذ بدون أخطاء")
finally:
    print("يُنفَّذ دائمًا")

raise ValueError("رسالة الخطأ")

class CustomError(Exception):
    pass

# ────────────────────────────────────────────────────────────────────────
# [012] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
class Animal:
    species_count = 0                 # خاصية صنف (Class Attribute)

    def __init__(self, name: str) -> None:
        self.name = name              # خاصية كائن (Instance Attribute)
        Animal.species_count += 1

    def speak(self) -> str:
        return f"{self.name} يصدر صوتًا"

    def __str__(self) -> str:
        return f"Animal({self.name})"

    def __repr__(self) -> str:
        return f"Animal(name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Animal):
            return NotImplemented
        return self.name == other.name

class Dog(Animal):                    # الوراثة
    def speak(self) -> str:           # تجاوز (Override)
        return f"{self.name} ينبح!"

# Dataclass
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list[str] = field(default_factory=list)

# Property
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

# ────────────────────────────────────────────────────────────────────────
# [013] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
# Decorator
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("قبل التنفيذ")
        result = func(*args, **kwargs)
        print("بعد التنفيذ")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("مرحبًا")

# Generator
def count_up(n: int):
    for i in range(n):
        yield i

# Closure
def make_multiplier(factor: int):
    def multiply(x: int) -> int:
        return x * factor
    return multiply

double = make_multiplier(2)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
from typing import TypeVar, Generic
from collections.abc import Callable, Iterator, Iterable

x: int
x: str | None                    # Union (بايثون 3.10+)
x: list[int]
x: dict[str, int]
x: tuple[int, str]
x: tuple[int, ...]               # عدد غير محدد من نفس النوع
x: Callable[[int, int], int]     # دالة تأخذ int,int وتُعيد int
x: Iterator[int]

T = TypeVar("T")
class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

# ────────────────────────────────────────────────────────────────────────
# [015] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

p = Path("data.txt")
p.write_text("محتوى", encoding="utf-8")
content = p.read_text(encoding="utf-8")
p.exists()
p.is_file()
p.mkdir(parents=True, exist_ok=True)

with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ────────────────────────────────────────────────────────────────────────
# [016] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import json

json.dumps(data, ensure_ascii=False, indent=2)   # كائن → نص JSON
json.loads(json_string)                          # نص JSON → كائن
Path("data.json").write_text(json.dumps(data))

# ────────────────────────────────────────────────────────────────────────
# [017] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import csv

with open("file.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()
    writer.writerow({"name": "علي", "age": 30})

# ────────────────────────────────────────────────────────────────────────
# [018] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def fetch(url: str) -> str:
    await asyncio.sleep(1)
    return f"data from {url}"

async def main() -> None:
    result = await fetch("url")
    results = await asyncio.gather(fetch("a"), fetch("b"))

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [019] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = Query(None)) -> dict:
    return {"item_id": item_id, "q": q}

@app.post("/items")
def create_item(item: Item) -> Item:
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item) -> dict:
    return {"item_id": item_id, **item.model_dump()}

@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict:
    if item_id not in db:
        raise HTTPException(status_code=404, detail="غير موجود")
    return {"message": "تم الحذف"}

# ────────────────────────────────────────────────────────────────────────
# [020] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# NumPy
arr = np.array([1, 2, 3])
arr.mean(), arr.sum(), arr.std()

# pandas - القراءة
df = pd.read_csv("file.csv")
df.head()
df.info()
df.describe()

# الاختيار
df["column"]
df.loc[0, "column"]        # بالتسمية
df.iloc[0, 0]               # بالموضع
df[df["age"] > 18]          # فلترة

# المعالجة
df.dropna()
df.fillna(0)
df.groupby("category")["sales"].sum()
df.sort_values("price", ascending=False)
df.merge(other_df, on="id")
pd.pivot_table(df, values="sales", index="region", aggfunc="sum")

# ────────────────────────────────────────────────────────────────────────
# [021] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.bar(categories, values)
plt.scatter(x, y)
plt.hist(data, bins=20)
plt.title("العنوان")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("chart.png")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [022] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipeline.fit(X_train, y_train)

import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

# ────────────────────────────────────────────────────────────────────────
# [023] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.argument("name")
@click.option("--greeting", default="مرحبًا", help="نص الترحيب")
@click.option("--loud", is_flag=True, help="بأحرف كبيرة")
def hello(name: str, greeting: str, loud: bool) -> None:
    message = f"{greeting}, {name}!"
    click.echo(message.upper() if loud else message)

@click.group()
def cli(): ...

@cli.command()
def sub_command(): ...

if __name__ == "__main__":
    hello()

# ────────────────────────────────────────────────────────────────────────
# [024] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
import pytest

def test_simple():
    assert 1 + 1 == 2

def test_exception():
    with pytest.raises(ValueError):
        int("abc")

def test_float():
    assert 0.1 + 0.2 == pytest.approx(0.3)

@pytest.fixture
def sample_data():
    return {"name": "test"}

def test_with_fixture(sample_data):
    assert sample_data["name"] == "test"

@pytest.mark.parametrize("input,expected", [(1, 1), (2, 4), (3, 9)])
def test_square(input, expected):
    assert input ** 2 == expected

# ────────────────────────────────────────────────────────────────────────
# [025] dist/markdown/appendix-c.md
# ────────────────────────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock

@patch("module.requests.get")
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"key": "value"}
    result = my_function()
    mock_get.assert_called_once()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# التعريف مع Type Hints
name: str = "محمد"
age: int = 30
price: float = 19.99
is_active: bool = True
data: None = None

# تحويل الأنواع
int("42")          # 42
float("3.14")      # 3.14
str(100)           # "100"
bool(0)            # False
list("abc")        # ['a', 'b', 'c']

# فحص النوع
type(x)             # النوع الفعلي
isinstance(x, int)  # فحص إن كان من نوع معيّن

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
+  -  *  /       # عمليات أساسية (/ تُنتج float دائمًا)
//              # قسمة صحيحة (floor division)
# %               # باقي القسمة
**              # قوة (أس)

==  !=  <  >  <=  >=    # مقارنة
and  or  not             # منطقية

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
name, age = "سارة", 25
f"{name} عمرها {age}"
f"{price:.2f}"           # تنسيق عشري بمنزلتين
f"{number:,}"            # فاصلة آلاف
f"{value:>10}"           # محاذاة يمين بعرض 10
f"{value=}"              # يطبع "value=10" (مفيد للتنقيح)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
s = "Hello World"

s.upper()           # "HELLO WORLD"
s.lower()           # "hello world"
s.strip()           # إزالة المسافات من الطرفين
s.replace("a", "b") # استبدال
s.split(" ")        # تقسيم لقائمة
"-".join(["a","b"]) # دمج قائمة لنص
s.startswith("He")  # True
s.endswith("ld")    # True
s.find("World")     # 6 (أو -1 إن لم يوجد)
len(s)              # طول النص
s[0]                # أول حرف
s[-1]               # آخر حرف
s[0:5]              # تقطيع (Slicing)
s[::-1]             # عكس النص

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
lst = [1, 2, 3]

lst.append(4)        # إضافة في النهاية
lst.insert(0, 0)      # إضافة في موضع محدد
lst.remove(2)         # إزالة أول قيمة مطابقة
lst.pop()             # إزالة وإرجاع آخر عنصر
lst.pop(0)            # إزالة وإرجاع عنصر بموضع محدد
lst.sort()            # ترتيب في المكان
sorted(lst)           # ترتيب في نسخة جديدة
lst.reverse()         # عكس في المكان
len(lst)              # الطول
lst[1:3]              # تقطيع
2 in lst              # فحص الوجود

# List Comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
d = {"name": "علي", "age": 30}

d["name"]              # قراءة
d["city"] = "الرياض"   # إضافة/تعديل
d.get("key", default)  # قراءة آمنة بقيمة افتراضية
d.keys()                # المفاتيح
d.values()              # القيم
d.items()               # أزواج (مفتاح، قيمة)
del d["age"]            # حذف
"name" in d             # فحص وجود مفتاح

# Dict Comprehension
squares = {x: x**2 for x in range(5)}

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
s = {1, 2, 3}

s.add(4)
s.remove(2)
s1 | s2    # اتحاد
s1 & s2    # تقاطع
s1 - s2    # فرق
s1 ^ s2    # فرق متماثل

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
t = (1, 2, 3)
a, b, c = t          # فك التغليف (Unpacking)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# شرطية
if condition:
    ...
elif other_condition:
    ...
else:
    ...

# التعبير الشرطي (Ternary)
result = "نعم" if condition else "لا"

# حلقة for
for item in iterable:
    ...
for i in range(10):          # 0 → 9
for i, item in enumerate(lst):  # مع الفهرس
for a, b in zip(list1, list2):  # تكرار متوازٍ

# حلقة while
while condition:
    ...

# كلمات تحكم
break       # إيقاف الحلقة
continue    # تخطي الدورة الحالية
pass        # عدم فعل شيء (حجز مكان)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "مرحبًا") -> str:
    """Docstring يشرح الدالة."""
    return f"{greeting}, {name}!"

# وسائط متغيرة
def total(*args: int) -> int:
    return sum(args)

def config(**kwargs: str) -> None:
    for key, value in kwargs.items():
        print(key, value)

# Lambda
square = lambda x: x ** 2

# Type Hints شائعة
def process(items: list[str]) -> dict[str, int]: ...
def maybe(value: int | None = None) -> bool: ...

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except ValueError as e:
    print(f"خطأ: {e}")
except (TypeError, KeyError):
    print("أحد هذين الخطأين")
else:
    print("نُفِّذ بدون أخطاء")
finally:
    print("يُنفَّذ دائمًا")

raise ValueError("رسالة الخطأ")

class CustomError(Exception):
    pass

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class Animal:
    species_count = 0                 # خاصية صنف (Class Attribute)

    def __init__(self, name: str) -> None:
        self.name = name              # خاصية كائن (Instance Attribute)
        Animal.species_count += 1

    def speak(self) -> str:
        return f"{self.name} يصدر صوتًا"

    def __str__(self) -> str:
        return f"Animal({self.name})"

    def __repr__(self) -> str:
        return f"Animal(name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Animal):
            return NotImplemented
        return self.name == other.name

class Dog(Animal):                    # الوراثة
    def speak(self) -> str:           # تجاوز (Override)
        return f"{self.name} ينبح!"

# Dataclass
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list[str] = field(default_factory=list)

# Property
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# Decorator
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("قبل التنفيذ")
        result = func(*args, **kwargs)
        print("بعد التنفيذ")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("مرحبًا")

# Generator
def count_up(n: int):
    for i in range(n):
        yield i

# Closure
def make_multiplier(factor: int):
    def multiply(x: int) -> int:
        return x * factor
    return multiply

double = make_multiplier(2)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from typing import TypeVar, Generic
from collections.abc import Callable, Iterator, Iterable

x: int
x: str | None                    # Union (بايثون 3.10+)
x: list[int]
x: dict[str, int]
x: tuple[int, str]
x: tuple[int, ...]               # عدد غير محدد من نفس النوع
x: Callable[[int, int], int]     # دالة تأخذ int,int وتُعيد int
x: Iterator[int]

T = TypeVar("T")
class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

p = Path("data.txt")
p.write_text("محتوى", encoding="utf-8")
content = p.read_text(encoding="utf-8")
p.exists()
p.is_file()
p.mkdir(parents=True, exist_ok=True)

with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import json

json.dumps(data, ensure_ascii=False, indent=2)   # كائن → نص JSON
json.loads(json_string)                          # نص JSON → كائن
Path("data.json").write_text(json.dumps(data))

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import csv

with open("file.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()
    writer.writerow({"name": "علي", "age": 30})

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def fetch(url: str) -> str:
    await asyncio.sleep(1)
    return f"data from {url}"

async def main() -> None:
    result = await fetch("url")
    results = await asyncio.gather(fetch("a"), fetch("b"))

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = Query(None)) -> dict:
    return {"item_id": item_id, "q": q}

@app.post("/items")
def create_item(item: Item) -> Item:
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item) -> dict:
    return {"item_id": item_id, **item.model_dump()}

@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict:
    if item_id not in db:
        raise HTTPException(status_code=404, detail="غير موجود")
    return {"message": "تم الحذف"}

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# NumPy
arr = np.array([1, 2, 3])
arr.mean(), arr.sum(), arr.std()

# pandas - القراءة
df = pd.read_csv("file.csv")
df.head()
df.info()
df.describe()

# الاختيار
df["column"]
df.loc[0, "column"]        # بالتسمية
df.iloc[0, 0]               # بالموضع
df[df["age"] > 18]          # فلترة

# المعالجة
df.dropna()
df.fillna(0)
df.groupby("category")["sales"].sum()
df.sort_values("price", ascending=False)
df.merge(other_df, on="id")
pd.pivot_table(df, values="sales", index="region", aggfunc="sum")

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.bar(categories, values)
plt.scatter(x, y)
plt.hist(data, bins=20)
plt.title("العنوان")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("chart.png")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipeline.fit(X_train, y_train)

import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
import click

@click.command()
@click.argument("name")
@click.option("--greeting", default="مرحبًا", help="نص الترحيب")
@click.option("--loud", is_flag=True, help="بأحرف كبيرة")
def hello(name: str, greeting: str, loud: bool) -> None:
    message = f"{greeting}, {name}!"
    click.echo(message.upper() if loud else message)

@click.group()
def cli(): ...

@cli.command()
def sub_command(): ...

if __name__ == "__main__":
    hello()

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
import pytest

def test_simple():
    assert 1 + 1 == 2

def test_exception():
    with pytest.raises(ValueError):
        int("abc")

def test_float():
    assert 0.1 + 0.2 == pytest.approx(0.3)

@pytest.fixture
def sample_data():
    return {"name": "test"}

def test_with_fixture(sample_data):
    assert sample_data["name"] == "test"

@pytest.mark.parametrize("input,expected", [(1, 1), (2, 4), (3, 9)])
def test_square(input, expected):
    assert input ** 2 == expected

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-c-appendix-c.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock

@patch("module.requests.get")
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"key": "value"}
    result = my_function()
    mock_get.assert_called_once()
