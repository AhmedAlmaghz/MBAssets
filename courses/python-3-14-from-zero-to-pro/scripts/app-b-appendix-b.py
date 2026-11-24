#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملحق ب: حلول جميع التمارين
Python 3.14 — From Zero to Pro — B

Book slug : python-3-14-from-zero-to-pro
Chapter   : app-b-appendix-b (lesson b)
Code files: 54 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/app-b-appendix-b.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملحق ب: حلول جميع التمارين
Python 3.14 — From Zero to Pro — B

Book slug : python-3-14-from-zero-to-pro
Chapter   : app-b-appendix-b (lesson b)
Code files: 53 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
name: str = "سارة"
age: int = 25
height: float = 1.65

print(f"اسمي {name}، عمري {age} سنة، وطولي {height} متر.")

# ────────────────────────────────────────────────────────────────────────
# [002] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
a: int = 5
b: int = 10

a, b = b, a  # الطريقة البايثونية لتبديل القيم مباشرة

print(f"a = {a}, b = {b}")  # a = 10, b = 5

# ────────────────────────────────────────────────────────────────────────
# [003] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
age_years: int = 25
days_per_year: float = 365.25  # لمراعاة السنوات الكبيسة تقريبيًا

age_in_days: float = age_years * days_per_year
print(f"العمر التقريبي بالأيام: {age_in_days:.0f} يوم")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
name = "Ahmed"
print(name.upper())   # AHMED
print(name.lower())   # ahmed

# ────────────────────────────────────────────────────────────────────────
# [005] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
sentence = "تعلم بايثون ممتع ومفيد"
word = "ممتع"

if word in sentence:
    print(f"الكلمة '{word}' موجودة في الجملة")
else:
    print(f"الكلمة '{word}' غير موجودة")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
sentence = "Python is powerful"
words = sentence.split()
reversed_words = [word[::-1] for word in words]
result = " ".join(reversed_words)

print(result)  # nohtyP si lufrewop

# ────────────────────────────────────────────────────────────────────────
# [007] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
length: float = 8.5
width: float = 4.2

area = length * width
perimeter = 2 * (length + width)

print(f"المساحة: {area} | المحيط: {perimeter}")

# ────────────────────────────────────────────────────────────────────────
# [008] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
fahrenheit: float = 98.6
celsius = (fahrenheit - 32) * 5 / 9

print(f"{fahrenheit}°F = {celsius:.2f}°C")

# ────────────────────────────────────────────────────────────────────────
# [009] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
principal: float = 1000.0
annual_rate: float = 0.05
years: int = 10

final_amount = principal * (1 + annual_rate) ** years
print(f"المبلغ بعد {years} سنوات: {final_amount:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
number = int(input("أدخل رقمًا: "))

if number % 2 == 0:
    print("الرقم زوجي")
else:
    print("الرقم فردي")

# ────────────────────────────────────────────────────────────────────────
# [011] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
score = int(input("أدخل الدرجة: "))

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"التقدير: {grade}")

# ────────────────────────────────────────────────────────────────────────
# [012] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
a, b, c = 5, 5, 8

if a == b == c:
    triangle_type = "متساوي الأضلاع"
elif a == b or b == c or a == c:
    triangle_type = "متساوي الساقين"
else:
    triangle_type = "مختلف الأضلاع"

# التحقق من صحة المثلث أولاً
if a + b > c and a + c > b and b + c > a:
    print(f"نوع المثلث: {triangle_type}")
else:
    print("هذه الأطوال لا تُشكّل مثلثًا صحيحًا")

# ────────────────────────────────────────────────────────────────────────
# [013] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
number = 7

for i in range(1, 11):
    print(f"{number} × {i} = {number * i}")

# ────────────────────────────────────────────────────────────────────────
# [014] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
total = sum(n for n in range(1, 101) if n % 2 != 0)
print(f"مجموع الأعداد الفردية: {total}")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
rows = 5

for i in range(1, rows + 1):
    print("*" * i)

# شكل هرمي متناظر
for i in range(1, rows + 1):
    spaces = " " * (rows - i)
    stars = "*" * (2 * i - 1)
    print(spaces + stars)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
fruits = ["تفاح", "موز", "برتقال", "عنب"]

for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 3, 2, 3, 1, 5, 4, 2]

seen = set()
unique_numbers = []
for n in numbers:
    if n not in seen:
        seen.add(n)
        unique_numbers.append(n)

print(unique_numbers)  # [1, 3, 2, 5, 4]

# ────────────────────────────────────────────────────────────────────────
# [018] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
def merge_sorted(list1: list[int], list2: list[int]) -> list[int]:
    merged = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

result = merge_sorted([1, 3, 5, 7], [2, 4, 6, 8])
print(result)  # [1, 2, 3, 4, 5, 6, 7, 8]

# ────────────────────────────────────────────────────────────────────────
# [019] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
student = {
    "name": "ليلى",
    "age": 20,
    "major": "هندسة برمجيات",
}

for key, value in student.items():
    print(f"{key}: {value}")

# ────────────────────────────────────────────────────────────────────────
# [020] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
text = "python is great python is fun"
words = text.split()

word_count: dict[str, int] = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

print(word_count)  # {'python': 2, 'is': 2, 'great': 1, 'fun': 1}

# ────────────────────────────────────────────────────────────────────────
# [021] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

common = set_a & set_b
only_in_a = set_a - set_b
only_in_b = set_b - set_a
all_unique = set_a ^ set_b  # الاختلاف المتماثل

print(f"مشترك: {common}")
print(f"فقط في أ: {only_in_a}")
print(f"فقط في ب: {only_in_b}")
print(f"غير مشترك بينهما: {all_unique}")

# ────────────────────────────────────────────────────────────────────────
# [022] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import math

def circle_area(radius: float) -> float:
    return math.pi * radius ** 2

print(f"{circle_area(5):.2f}")  # 78.54

# ────────────────────────────────────────────────────────────────────────
# [023] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 50) if is_prime(n)]
print(primes)

# ────────────────────────────────────────────────────────────────────────
# [024] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

result = [fibonacci(i) for i in range(15)]
print(result)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("خطأ: لا يمكن القسمة على صفر")
        return None

print(safe_divide(10, 2))   # 5.0
print(safe_divide(10, 0))   # خطأ: ... ثم None

# ────────────────────────────────────────────────────────────────────────
# [026] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
def get_valid_number() -> int:
    while True:
        try:
            return int(input("أدخل رقمًا صحيحًا: "))
        except ValueError:
            print("إدخال غير صالح، حاول مرة أخرى")

# number = get_valid_number()

# ────────────────────────────────────────────────────────────────────────
# [027] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    def __init__(self, balance: float, amount: float) -> None:
        self.balance = balance
        self.amount = amount
        super().__init__(
            f"رصيد غير كافٍ: الرصيد {balance}, المطلوب سحبه {amount}"
        )

def withdraw(balance: float, amount: float) -> float:
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(100, 150)
except InsufficientFundsError as e:
    print(f"فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [028] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
pairs = [(1, 5), (2, 2), (3, 8), (4, 1)]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs)  # [(4, 1), (2, 2), (1, 5), (3, 8)]

# ────────────────────────────────────────────────────────────────────────
# [029] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from collections.abc import Iterator

def prime_generator() -> Iterator[int]:
    n = 2
    while True:
        if all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)):
            yield n
        n += 1

gen = prime_generator()
first_10_primes = [next(gen) for _ in range(10)]
print(first_10_primes)

# ────────────────────────────────────────────────────────────────────────
# [030] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

def timer(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} استغرقت {elapsed:.4f} ثانية")
        return result
    return wrapper

@timer
def slow_function() -> None:
    time.sleep(0.5)

slow_function()

# ────────────────────────────────────────────────────────────────────────
# [031] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
class Car:
    def __init__(self, brand: str, model: str, year: int) -> None:
        self.brand = brand
        self.model = model
        self.year = year

    def honk(self) -> str:
        return f"{self.brand} {self.model}: بيب بيب! 🚗"

    def __str__(self) -> str:
        return f"{self.year} {self.brand} {self.model}"

car = Car("Toyota", "Corolla", 2024)
print(car)
print(car.honk())

# ────────────────────────────────────────────────────────────────────────
# [032] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
class ElectricCar(Car):
    def __init__(
        self, brand: str, model: str, year: int, battery_range: int
    ) -> None:
        super().__init__(brand, model, year)
        self.battery_range = battery_range

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (مدى البطارية: {self.battery_range} كم)"

tesla = ElectricCar("Tesla", "Model 3", 2024, 500)
print(tesla)

# ────────────────────────────────────────────────────────────────────────
# [033] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

shapes: list[Shape] = [Circle(5), Rectangle(4, 6)]

for shape in shapes:
    print(f"{type(shape).__name__}: مساحة={shape.area():.2f}, محيط={shape.perimeter():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [034] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Order:
    item: str
    quantity: int
    price: float

def total_revenue(orders: list[Order]) -> float:
    return sum(order.quantity * order.price for order in orders)

orders = [Order("كتاب", 3, 20.0), Order("قلم", 10, 2.5)]
print(total_revenue(orders))  # 85.0

# ────────────────────────────────────────────────────────────────────────
# [035] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0

int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)
print(int_stack.pop())  # 2

# ────────────────────────────────────────────────────────────────────────
# [036] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import csv
from pathlib import Path

csv_path = Path("students.csv")

ages: list[int] = []
with csv_path.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ages.append(int(row["age"]))

average_age = sum(ages) / len(ages) if ages else 0
print(f"متوسط الأعمار: {average_age:.1f}")

# ────────────────────────────────────────────────────────────────────────
# [037] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Contact:
    name: str
    phone: str

contacts = [Contact("أحمد", "0501234567"), Contact("سارة", "0559876543")]

# الحفظ
data = [asdict(c) for c in contacts]
Path("contacts.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
)

# الاسترجاع
loaded_data = json.loads(Path("contacts.json").read_text(encoding="utf-8"))
loaded_contacts = [Contact(**item) for item in loaded_data]
print(loaded_contacts)

# ────────────────────────────────────────────────────────────────────────
# [038] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def fetch_data(source: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"بيانات من {source}"

async def main() -> None:
    results = await asyncio.gather(
        fetch_data("API-1", 1),
        fetch_data("API-2", 2),
        fetch_data("API-3", 0.5),
    )
    print(results)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [039] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def limited_task(semaphore: asyncio.Semaphore, task_id: int) -> None:
    async with semaphore:
        print(f"بدء المهمة {task_id}")
        await asyncio.sleep(1)
        print(f"انتهت المهمة {task_id}")

async def main() -> None:
    semaphore = asyncio.Semaphore(3)  # 3 مهام متزامنة كحد أقصى
    tasks = [limited_task(semaphore, i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [040] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException

app = FastAPI()
tasks_db: dict[int, dict] = {1: {"title": "تعلم FastAPI"}}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    del tasks_db[task_id]
    return {"message": "تم الحذف بنجاح"}

# ────────────────────────────────────────────────────────────────────────
# [041] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, Query

app = FastAPI()
all_tasks = [{"id": i, "title": f"مهمة {i}"} for i in range(1, 101)]

@app.get("/tasks")
def list_tasks(
    page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)
) -> dict:
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": len(all_tasks),
        "items": all_tasks[start:end],
    }

# ────────────────────────────────────────────────────────────────────────
# [042] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv("sales.csv")
top_5 = df.nlargest(5, "sales_amount")
print(top_5[["product", "sales_amount"]])

# ────────────────────────────────────────────────────────────────────────
# [043] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sales.csv", parse_dates=["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["sales_amount"].sum()

colors = ["orange" if v == monthly.max() else "steelblue" for v in monthly]

plt.figure(figsize=(10, 5))
plt.bar(monthly.index.astype(str), monthly.values, color=colors)
plt.title("المبيعات الشهرية")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales.png")

# ────────────────────────────────────────────────────────────────────────
# [044] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com", timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

links = [a["href"] for a in soup.find_all("a", href=True)]
print(links)

# ────────────────────────────────────────────────────────────────────────
# [045] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import time
import requests
from bs4 import BeautifulSoup

all_titles: list[str] = []

for page in range(1, 4):
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    titles = [h3.a["title"] for h3 in soup.select("h3")]
    all_titles.extend(titles)
    time.sleep(1)  # احترام الخادم

print(f"إجمالي الكتب المستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [046] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"الدقة: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [047] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"{name}: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [048] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import click
from pathlib import Path

@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def wordcount(file_path: str) -> None:
    """يحسب عدد الأسطر والكلمات والأحرف في ملف نصي."""
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    words = text.split()

    click.echo(f"الأسطر: {len(lines)}")
    click.echo(f"الكلمات: {len(words)}")
    click.echo(f"الأحرف: {len(text)}")

if __name__ == "__main__":
    wordcount()

# ────────────────────────────────────────────────────────────────────────
# [049] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import click

@click.group()
def cli() -> None:
    """أداة إدارة مهام بسيطة."""
    pass

@cli.command()
@click.argument("title")
def add(title: str) -> None:
    """إضافة مهمة جديدة."""
    click.echo(f"تمت إضافة: {title}")

@cli.command()
def list() -> None:
    """عرض كل المهام."""
    click.echo("عرض المهام...")

if __name__ == "__main__":
    cli()

# ────────────────────────────────────────────────────────────────────────
# [050] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
# قبل: دالة واحدة تفعل كل شيء
def process_order_bad(order: dict) -> None:
    # تحقق + حساب + طباعة + حفظ، كل هذا في دالة واحدة
    ...

# بعد: تقسيم حسب المسؤولية
def validate_order(order: dict) -> bool:
    return order.get("quantity", 0) > 0

def calculate_total(order: dict) -> float:
    return order["quantity"] * order["price"]

def save_order(order: dict, total: float) -> None:
    print(f"تم حفظ الطلب بمبلغ {total}")

def process_order(order: dict) -> None:
    if not validate_order(order):
        raise ValueError("طلب غير صالح")
    total = calculate_total(order)
    save_order(order, total)

# ────────────────────────────────────────────────────────────────────────
# [051] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
import pytest

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

@pytest.mark.parametrize(
    "number, expected",
    [(2, True), (4, False), (17, True), (1, False), (0, False)],
)
def test_is_prime(number: int, expected: bool) -> None:
    assert is_prime(number) == expected

# ────────────────────────────────────────────────────────────────────────
# [052] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock
import requests

def get_user_name(user_id: int) -> str:
    response = requests.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    return response.json()["name"]

@patch("requests.get")
def test_get_user_name(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "أحمد"}
    mock_get.return_value = mock_response

    result = get_user_name(1)

    assert result == "أحمد"
    mock_get.assert_called_once_with("https://api.example.com/users/1")

# ────────────────────────────────────────────────────────────────────────
# [053] dist/markdown/appendix-b.md
# ────────────────────────────────────────────────────────────────────────
# src/math_helpers/__init__.py
from math_helpers.core import is_prime, fibonacci

__all__ = ["is_prime", "fibonacci"]
__version__ = "0.1.0"

# src/math_helpers/core.py
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

def fibonacci(n: int) -> list[int]:
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
name: str = "سارة"
age: int = 25
height: float = 1.65

print(f"اسمي {name}، عمري {age} سنة، وطولي {height} متر.")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
a: int = 5
b: int = 10

a, b = b, a  # الطريقة البايثونية لتبديل القيم مباشرة

print(f"a = {a}, b = {b}")  # a = 10, b = 5

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
age_years: int = 25
days_per_year: float = 365.25  # لمراعاة السنوات الكبيسة تقريبيًا

age_in_days: float = age_years * days_per_year
print(f"العمر التقريبي بالأيام: {age_in_days:.0f} يوم")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
name = "Ahmed"
print(name.upper())   # AHMED
print(name.lower())   # ahmed

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
sentence = "تعلم بايثون ممتع ومفيد"
word = "ممتع"

if word in sentence:
    print(f"الكلمة '{word}' موجودة في الجملة")
else:
    print(f"الكلمة '{word}' غير موجودة")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
sentence = "Python is powerful"
words = sentence.split()
reversed_words = [word[::-1] for word in words]
result = " ".join(reversed_words)

print(result)  # nohtyP si lufrewop

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
length: float = 8.5
width: float = 4.2

area = length * width
perimeter = 2 * (length + width)

print(f"المساحة: {area} | المحيط: {perimeter}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
fahrenheit: float = 98.6
celsius = (fahrenheit - 32) * 5 / 9

print(f"{fahrenheit}°F = {celsius:.2f}°C")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
principal: float = 1000.0
annual_rate: float = 0.05
years: int = 10

final_amount = principal * (1 + annual_rate) ** years
print(f"المبلغ بعد {years} سنوات: {final_amount:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
number = int(input("أدخل رقمًا: "))

if number % 2 == 0:
    print("الرقم زوجي")
else:
    print("الرقم فردي")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
score = int(input("أدخل الدرجة: "))

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"التقدير: {grade}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
a, b, c = 5, 5, 8

if a == b == c:
    triangle_type = "متساوي الأضلاع"
elif a == b or b == c or a == c:
    triangle_type = "متساوي الساقين"
else:
    triangle_type = "مختلف الأضلاع"

# التحقق من صحة المثلث أولاً
if a + b > c and a + c > b and b + c > a:
    print(f"نوع المثلث: {triangle_type}")
else:
    print("هذه الأطوال لا تُشكّل مثلثًا صحيحًا")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
number = 7

for i in range(1, 11):
    print(f"{number} × {i} = {number * i}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
total = sum(n for n in range(1, 101) if n % 2 != 0)
print(f"مجموع الأعداد الفردية: {total}")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
rows = 5

for i in range(1, rows + 1):
    print("*" * i)

# شكل هرمي متناظر
for i in range(1, rows + 1):
    spaces = " " * (rows - i)
    stars = "*" * (2 * i - 1)
    print(spaces + stars)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
fruits = ["تفاح", "موز", "برتقال", "عنب"]

for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 3, 2, 3, 1, 5, 4, 2]

seen = set()
unique_numbers = []
for n in numbers:
    if n not in seen:
        seen.add(n)
        unique_numbers.append(n)

print(unique_numbers)  # [1, 3, 2, 5, 4]

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def merge_sorted(list1: list[int], list2: list[int]) -> list[int]:
    merged = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

result = merge_sorted([1, 3, 5, 7], [2, 4, 6, 8])
print(result)  # [1, 2, 3, 4, 5, 6, 7, 8]

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
student = {
    "name": "ليلى",
    "age": 20,
    "major": "هندسة برمجيات",
}

for key, value in student.items():
    print(f"{key}: {value}")

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
text = "python is great python is fun"
words = text.split()

word_count: dict[str, int] = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

print(word_count)  # {'python': 2, 'is': 2, 'great': 1, 'fun': 1}

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

common = set_a & set_b
only_in_a = set_a - set_b
only_in_b = set_b - set_a
all_unique = set_a ^ set_b  # الاختلاف المتماثل

print(f"مشترك: {common}")
print(f"فقط في أ: {only_in_a}")
print(f"فقط في ب: {only_in_b}")
print(f"غير مشترك بينهما: {all_unique}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import math

def circle_area(radius: float) -> float:
    return math.pi * radius ** 2

print(f"{circle_area(5):.2f}")  # 78.54

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 50) if is_prime(n)]
print(primes)

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

result = [fibonacci(i) for i in range(15)]
print(result)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("خطأ: لا يمكن القسمة على صفر")
        return None

print(safe_divide(10, 2))   # 5.0
print(safe_divide(10, 0))   # خطأ: ... ثم None

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def get_valid_number() -> int:
    while True:
        try:
            return int(input("أدخل رقمًا صحيحًا: "))
        except ValueError:
            print("إدخال غير صالح، حاول مرة أخرى")

# number = get_valid_number()

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    def __init__(self, balance: float, amount: float) -> None:
        self.balance = balance
        self.amount = amount
        super().__init__(
            f"رصيد غير كافٍ: الرصيد {balance}, المطلوب سحبه {amount}"
        )

def withdraw(balance: float, amount: float) -> float:
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(100, 150)
except InsufficientFundsError as e:
    print(f"فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
pairs = [(1, 5), (2, 2), (3, 8), (4, 1)]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs)  # [(4, 1), (2, 2), (1, 5), (3, 8)]

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from collections.abc import Iterator

def prime_generator() -> Iterator[int]:
    n = 2
    while True:
        if all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)):
            yield n
        n += 1

gen = prime_generator()
first_10_primes = [next(gen) for _ in range(10)]
print(first_10_primes)

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

def timer(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} استغرقت {elapsed:.4f} ثانية")
        return result
    return wrapper

@timer
def slow_function() -> None:
    time.sleep(0.5)

slow_function()

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
class Car:
    def __init__(self, brand: str, model: str, year: int) -> None:
        self.brand = brand
        self.model = model
        self.year = year

    def honk(self) -> str:
        return f"{self.brand} {self.model}: بيب بيب! 🚗"

    def __str__(self) -> str:
        return f"{self.year} {self.brand} {self.model}"

car = Car("Toyota", "Corolla", 2024)
print(car)
print(car.honk())

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
class ElectricCar(Car):
    def __init__(
        self, brand: str, model: str, year: int, battery_range: int
    ) -> None:
        super().__init__(brand, model, year)
        self.battery_range = battery_range

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (مدى البطارية: {self.battery_range} كم)"

tesla = ElectricCar("Tesla", "Model 3", 2024, 500)
print(tesla)

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

shapes: list[Shape] = [Circle(5), Rectangle(4, 6)]

for shape in shapes:
    print(f"{type(shape).__name__}: مساحة={shape.area():.2f}, محيط={shape.perimeter():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Order:
    item: str
    quantity: int
    price: float

def total_revenue(orders: list[Order]) -> float:
    return sum(order.quantity * order.price for order in orders)

orders = [Order("كتاب", 3, 20.0), Order("قلم", 10, 2.5)]
print(total_revenue(orders))  # 85.0

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0

int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)
print(int_stack.pop())  # 2

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
import csv
from pathlib import Path

csv_path = Path("students.csv")

ages: list[int] = []
with csv_path.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ages.append(int(row["age"]))

average_age = sum(ages) / len(ages) if ages else 0
print(f"متوسط الأعمار: {average_age:.1f}")

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Contact:
    name: str
    phone: str

contacts = [Contact("أحمد", "0501234567"), Contact("سارة", "0559876543")]

# الحفظ
data = [asdict(c) for c in contacts]
Path("contacts.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
)

# الاسترجاع
loaded_data = json.loads(Path("contacts.json").read_text(encoding="utf-8"))
loaded_contacts = [Contact(**item) for item in loaded_data]
print(loaded_contacts)

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def fetch_data(source: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"بيانات من {source}"

async def main() -> None:
    results = await asyncio.gather(
        fetch_data("API-1", 1),
        fetch_data("API-2", 2),
        fetch_data("API-3", 0.5),
    )
    print(results)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def limited_task(semaphore: asyncio.Semaphore, task_id: int) -> None:
    async with semaphore:
        print(f"بدء المهمة {task_id}")
        await asyncio.sleep(1)
        print(f"انتهت المهمة {task_id}")

async def main() -> None:
    semaphore = asyncio.Semaphore(3)  # 3 مهام متزامنة كحد أقصى
    tasks = [limited_task(semaphore, i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException

app = FastAPI()
tasks_db: dict[int, dict] = {1: {"title": "تعلم FastAPI"}}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    del tasks_db[task_id]
    return {"message": "تم الحذف بنجاح"}

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, Query

app = FastAPI()
all_tasks = [{"id": i, "title": f"مهمة {i}"} for i in range(1, 101)]

@app.get("/tasks")
def list_tasks(
    page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)
) -> dict:
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": len(all_tasks),
        "items": all_tasks[start:end],
    }

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv("sales.csv")
top_5 = df.nlargest(5, "sales_amount")
print(top_5[["product", "sales_amount"]])

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sales.csv", parse_dates=["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["sales_amount"].sum()

colors = ["orange" if v == monthly.max() else "steelblue" for v in monthly]

plt.figure(figsize=(10, 5))
plt.bar(monthly.index.astype(str), monthly.values, color=colors)
plt.title("المبيعات الشهرية")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales.png")

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com", timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

links = [a["href"] for a in soup.find_all("a", href=True)]
print(links)

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
import time
import requests
from bs4 import BeautifulSoup

all_titles: list[str] = []

for page in range(1, 4):
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    titles = [h3.a["title"] for h3 in soup.select("h3")]
    all_titles.extend(titles)
    time.sleep(1)  # احترام الخادم

print(f"إجمالي الكتب المستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"الدقة: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"{name}: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
import click
from pathlib import Path

@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def wordcount(file_path: str) -> None:
    """يحسب عدد الأسطر والكلمات والأحرف في ملف نصي."""
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    words = text.split()

    click.echo(f"الأسطر: {len(lines)}")
    click.echo(f"الكلمات: {len(words)}")
    click.echo(f"الأحرف: {len(text)}")

if __name__ == "__main__":
    wordcount()

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
import click

@click.group()
def cli() -> None:
    """أداة إدارة مهام بسيطة."""
    pass

@cli.command()
@click.argument("title")
def add(title: str) -> None:
    """إضافة مهمة جديدة."""
    click.echo(f"تمت إضافة: {title}")

@cli.command()
def list() -> None:
    """عرض كل المهام."""
    click.echo("عرض المهام...")

if __name__ == "__main__":
    cli()

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
# قبل: دالة واحدة تفعل كل شيء
def process_order_bad(order: dict) -> None:
    # تحقق + حساب + طباعة + حفظ، كل هذا في دالة واحدة
    ...

# بعد: تقسيم حسب المسؤولية
def validate_order(order: dict) -> bool:
    return order.get("quantity", 0) > 0

def calculate_total(order: dict) -> float:
    return order["quantity"] * order["price"]

def save_order(order: dict, total: float) -> None:
    print(f"تم حفظ الطلب بمبلغ {total}")

def process_order(order: dict) -> None:
    if not validate_order(order):
        raise ValueError("طلب غير صالح")
    total = calculate_total(order)
    save_order(order, total)

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
import pytest

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

@pytest.mark.parametrize(
    "number, expected",
    [(2, True), (4, False), (17, True), (1, False), (0, False)],
)
def test_is_prime(number: int, expected: bool) -> None:
    assert is_prime(number) == expected

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock
import requests

def get_user_name(user_id: int) -> str:
    response = requests.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    return response.json()["name"]

@patch("requests.get")
def test_get_user_name(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "أحمد"}
    mock_get.return_value = mock_response

    result = get_user_name(1)

    assert result == "أحمد"
    mock_get.assert_called_once_with("https://api.example.com/users/1")

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/app-b-appendix-b.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
# src/math_helpers/__init__.py
from math_helpers.core import is_prime, fibonacci

__all__ = ["is_prime", "fibonacci"]
__version__ = "0.1.0"

# src/math_helpers/core.py
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

def fibonacci(n: int) -> list[int]:
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]
