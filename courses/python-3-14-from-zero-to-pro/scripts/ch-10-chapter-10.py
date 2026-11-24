#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال - تنظيم الكود
Python 3.14 — From Zero to Pro — Chapter 10 — Solid Foundations

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 89 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-10-chapter-10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال - تنظيم الكود
Python 3.14 — From Zero to Pro — Chapter 10 — Solid Foundations

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 88 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-10-chapter-10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال - تنظيم الكود
Python 3.14 — From Zero to Pro — Chapter 10 — Solid Foundations

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 87 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # خطأ شائع
    def add(a: int, b: int) -> None:
        print(a + b)  # يطبع فقط، لا يُرجع!

    result = add(3, 5)  # result = None وليس 8!
    print(result + 1)  # 💥 TypeError: unsupported operand type(s)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-10.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-10.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-10.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-10.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-10.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-10.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-10.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # استخدام * لإجبار ما بعدها أن تكون keyword-only
    def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
        return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

    # print(create_order("Laptop", 2))
    # 💥 TypeError: يجب تمرير quantity بالاسم

    print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

    # استخدام / لإجبار ما قبلها أن تكون positional-only
    def divide(a: float, b: float, /) -> float:
        return a / b

    print(divide(10, 2))      # ✅ صحيح
    # print(divide(a=10, b=2))
    # 💥 TypeError: a و b يجب تمريرهما بالموضع فقط
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-10.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-10.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-10.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-10.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-10.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    x = 10  # متغير عام (Global)

    def my_function():
        y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
        print(x)  # يمكن قراءة المتغير العام من داخل الدالة
        print(y)

    my_function()
    print(x)  # 10 - متاح
    # print(y)
    # 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-10.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    counter = 0

    def increment():
        counter += 1  # 💥 UnboundLocalError!

    increment()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-10.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-10.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-10.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-10.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-10.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-10.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-10.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-10.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-10.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-10.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-10.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [033] dist/notebooks/chapter-10.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [034] dist/notebooks/chapter-10.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [035] dist/notebooks/chapter-10.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [036] dist/notebooks/chapter-10.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch10/10-001-calculate_circle_area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.1: لماذا تُعتبر الدوال نقطة تحوّل في رحلتك البرمجية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-001-calculate_circle_area.py
"""

# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch10/10-002-function_name.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-002-function_name.py
"""

def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch10/10-003-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-003-greet.py
"""

def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch10/10-004-calculate_bmi.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-004-calculate_bmi.py
"""

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch10/10-005-print_separator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-005-print_separator.py
"""

def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch10/10-006-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-006-add.py
"""

# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch10/10-007-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-007-add.py
"""

# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch10/10-008-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-008-greet.py
"""

def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch10/10-009-create_profile.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-009-create_profile.py
"""

def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch10/10-010-ex10_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-010-ex10_3.py
"""

# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch10/10-011-ex10_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-011-ex10_3.py
"""

# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch10/10-012-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-012-greet.py
"""

def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch10/10-013-add_item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-013-add_item.py
"""

# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch10/10-014-add_item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-014-add_item.py
"""

# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch10/10-015-calculate_sum.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-015-calculate_sum.py
"""

def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch10/10-016-create_user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-016-create_user.py
"""

def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch10/10-017-full_example.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-017-full_example.py
"""

def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch10/10-018-create_order.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-018-create_order.py
"""

# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch10/10-019-get_min_max.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-019-get_min_max.py
"""

def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch10/10-020-calculate_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-020-calculate_discount.py
"""

def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch10/10-021-calculate_discount_messy.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-021-calculate_discount_messy.py
"""

def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch10/10-022-log_message.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-022-log_message.py
"""

def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch10/10-023-my_function.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-023-my_function.py
"""

x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch10/10-024-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-024-increment.py
"""

counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch10/10-025-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-025-increment.py
"""

counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch10/10-026-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-026-increment.py
"""

# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch10/10-027-outer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-027-outer.py
"""

x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch10/10-028-add_tax.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-028-add_tax.py
"""

# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch10/10-029-modify_number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-029-modify_number.py
"""

def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch10/10-030-modify_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-030-modify_list.py
"""

def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch10/10-031-process_safely.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-031-process_safely.py
"""

def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch10/10-032-square.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-032-square.py
"""

def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [069] code/chapters/ch10/10-033-get_score.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-033-get_score.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [070] code/chapters/ch10/10-034-format_value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.8: Type Hints المتقدمة في الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-034-format_value.py
"""

# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [071] code/chapters/ch10/10-035-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-035-add.py
"""

import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [072] code/chapters/ch10/10-036-log_calculation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-036-log_calculation.py
"""

calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [073] code/chapters/ch10/10-037-is_even.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-037-is_even.py
"""

# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [074] code/chapters/ch10/10-038-is_valid_password.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-038-is_valid_password.py
"""

# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [075] code/chapters/ch10/10-039-memoize.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-039-memoize.py
"""

def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [076] code/chapters/ch10/10-040-modify.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.11: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-040-modify.py
"""

def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [077] code/chapters/ch10/10-041-func_name.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-041-func_name.py
"""

# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [078] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [079] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [080] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [081] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [082] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [083] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [084] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [085] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [086] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [087] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # خطأ شائع
    def add(a: int, b: int) -> None:
        print(a + b)  # يطبع فقط، لا يُرجع!

    result = add(3, 5)  # result = None وليس 8!
    print(result + 1)  # 💥 TypeError: unsupported operand type(s)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # استخدام * لإجبار ما بعدها أن تكون keyword-only
    def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
        return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

    # print(create_order("Laptop", 2))
    # 💥 TypeError: يجب تمرير quantity بالاسم

    print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

    # استخدام / لإجبار ما قبلها أن تكون positional-only
    def divide(a: float, b: float, /) -> float:
        return a / b

    print(divide(10, 2))      # ✅ صحيح
    # print(divide(a=10, b=2))
    # 💥 TypeError: a و b يجب تمريرهما بالموضع فقط
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    x = 10  # متغير عام (Global)

    def my_function():
        y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
        print(x)  # يمكن قراءة المتغير العام من داخل الدالة
        print(y)

    my_function()
    print(x)  # 10 - متاح
    # print(y)
    # 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    counter = 0

    def increment():
        counter += 1  # 💥 UnboundLocalError!

    increment()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.1: لماذا تُعتبر الدوال نقطة تحوّل في رحلتك البرمجية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-001-calculate_circle_area.py
"""

# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-002-function_name.py
"""

def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-003-greet.py
"""

def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-004-calculate_bmi.py
"""

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-005-print_separator.py
"""

def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-006-add.py
"""

# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-007-add.py
"""

# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-008-greet.py
"""

def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-009-create_profile.py
"""

def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-010-ex10_3.py
"""

# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-011-ex10_3.py
"""

# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-012-greet.py
"""

def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-013-add_item.py
"""

# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-014-add_item.py
"""

# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-015-calculate_sum.py
"""

def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-016-create_user.py
"""

def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-017-full_example.py
"""

def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-018-create_order.py
"""

# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-019-get_min_max.py
"""

def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-020-calculate_discount.py
"""

def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-021-calculate_discount_messy.py
"""

def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-022-log_message.py
"""

def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-023-my_function.py
"""

x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-024-increment.py
"""

counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-025-increment.py
"""

counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-026-increment.py
"""

# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-027-outer.py
"""

x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-028-add_tax.py
"""

# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-029-modify_number.py
"""

def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-030-modify_list.py
"""

def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-031-process_safely.py
"""

def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-032-square.py
"""

def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-033-get_score.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.8: Type Hints المتقدمة في الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-034-format_value.py
"""

# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-035-add.py
"""

import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-036-log_calculation.py
"""

calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-037-is_even.py
"""

# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-038-is_valid_password.py
"""

# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [076] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-039-memoize.py
"""

def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [077] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-76
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.11: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-040-modify.py
"""

def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [078] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-77
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-041-func_name.py
"""

# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [079] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-78
# ────────────────────────────────────────────────────────────────────────
def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [080] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-79
# ────────────────────────────────────────────────────────────────────────
# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [081] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-80
# ────────────────────────────────────────────────────────────────────────
# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [082] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-81
# ────────────────────────────────────────────────────────────────────────
# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [083] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-82
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [084] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-83
# ────────────────────────────────────────────────────────────────────────
# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [085] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-84
# ────────────────────────────────────────────────────────────────────────
x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [086] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-85
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [087] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-86
# ────────────────────────────────────────────────────────────────────────
import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [088] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-87
# ────────────────────────────────────────────────────────────────────────
# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال - تنظيم الكود
Python 3.14 — From Zero to Pro — Chapter 10 — Solid Foundations

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-10-chapter-10 (lesson 10)
Code files: 87 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # خطأ شائع
    def add(a: int, b: int) -> None:
        print(a + b)  # يطبع فقط، لا يُرجع!

    result = add(3, 5)  # result = None وليس 8!
    print(result + 1)  # 💥 TypeError: unsupported operand type(s)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-10.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-10.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-10.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-10.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-10.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-10.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-10.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # استخدام * لإجبار ما بعدها أن تكون keyword-only
    def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
        return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

    # print(create_order("Laptop", 2))
    # 💥 TypeError: يجب تمرير quantity بالاسم

    print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

    # استخدام / لإجبار ما قبلها أن تكون positional-only
    def divide(a: float, b: float, /) -> float:
        return a / b

    print(divide(10, 2))      # ✅ صحيح
    # print(divide(a=10, b=2))
    # 💥 TypeError: a و b يجب تمريرهما بالموضع فقط
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-10.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-10.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-10.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-10.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-10.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    x = 10  # متغير عام (Global)

    def my_function():
        y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
        print(x)  # يمكن قراءة المتغير العام من داخل الدالة
        print(y)

    my_function()
    print(x)  # 10 - متاح
    # print(y)
    # 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-10.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    counter = 0

    def increment():
        counter += 1  # 💥 UnboundLocalError!

    increment()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-10.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-10.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-10.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-10.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-10.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-10.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-10.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-10.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-10.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-10.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-10.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [033] dist/notebooks/chapter-10.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [034] dist/notebooks/chapter-10.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [035] dist/notebooks/chapter-10.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [036] dist/notebooks/chapter-10.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch10/10-001-calculate_circle_area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.1: لماذا تُعتبر الدوال نقطة تحوّل في رحلتك البرمجية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-001-calculate_circle_area.py
"""

# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch10/10-002-function_name.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-002-function_name.py
"""

def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch10/10-003-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-003-greet.py
"""

def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch10/10-004-calculate_bmi.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-004-calculate_bmi.py
"""

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch10/10-005-print_separator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-005-print_separator.py
"""

def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch10/10-006-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-006-add.py
"""

# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch10/10-007-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-007-add.py
"""

# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch10/10-008-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-008-greet.py
"""

def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch10/10-009-create_profile.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-009-create_profile.py
"""

def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch10/10-010-ex10_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-010-ex10_3.py
"""

# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch10/10-011-ex10_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-011-ex10_3.py
"""

# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch10/10-012-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-012-greet.py
"""

def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch10/10-013-add_item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-013-add_item.py
"""

# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch10/10-014-add_item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-014-add_item.py
"""

# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch10/10-015-calculate_sum.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-015-calculate_sum.py
"""

def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch10/10-016-create_user.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-016-create_user.py
"""

def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch10/10-017-full_example.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-017-full_example.py
"""

def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch10/10-018-create_order.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-018-create_order.py
"""

# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch10/10-019-get_min_max.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-019-get_min_max.py
"""

def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch10/10-020-calculate_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-020-calculate_discount.py
"""

def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch10/10-021-calculate_discount_messy.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-021-calculate_discount_messy.py
"""

def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch10/10-022-log_message.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-022-log_message.py
"""

def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch10/10-023-my_function.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-023-my_function.py
"""

x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch10/10-024-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-024-increment.py
"""

counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch10/10-025-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-025-increment.py
"""

counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch10/10-026-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-026-increment.py
"""

# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch10/10-027-outer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-027-outer.py
"""

x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch10/10-028-add_tax.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-028-add_tax.py
"""

# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch10/10-029-modify_number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-029-modify_number.py
"""

def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch10/10-030-modify_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-030-modify_list.py
"""

def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch10/10-031-process_safely.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-031-process_safely.py
"""

def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch10/10-032-square.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-032-square.py
"""

def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [069] code/chapters/ch10/10-033-get_score.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-033-get_score.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [070] code/chapters/ch10/10-034-format_value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.8: Type Hints المتقدمة في الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-034-format_value.py
"""

# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [071] code/chapters/ch10/10-035-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-035-add.py
"""

import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [072] code/chapters/ch10/10-036-log_calculation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-036-log_calculation.py
"""

calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [073] code/chapters/ch10/10-037-is_even.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-037-is_even.py
"""

# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [074] code/chapters/ch10/10-038-is_valid_password.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-038-is_valid_password.py
"""

# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [075] code/chapters/ch10/10-039-memoize.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-039-memoize.py
"""

def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [076] code/chapters/ch10/10-040-modify.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.11: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-040-modify.py
"""

def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [077] code/chapters/ch10/10-041-func_name.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-041-func_name.py
"""

# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [078] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [079] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [080] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [081] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [082] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [083] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [084] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [085] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [086] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [087] dist/markdown/chapter-10.md
# ────────────────────────────────────────────────────────────────────────
# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # خطأ شائع
    def add(a: int, b: int) -> None:
        print(a + b)  # يطبع فقط، لا يُرجع!

    result = add(3, 5)  # result = None وليس 8!
    print(result + 1)  # 💥 TypeError: unsupported operand type(s)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # استخدام * لإجبار ما بعدها أن تكون keyword-only
    def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
        return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

    # print(create_order("Laptop", 2))
    # 💥 TypeError: يجب تمرير quantity بالاسم

    print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

    # استخدام / لإجبار ما قبلها أن تكون positional-only
    def divide(a: float, b: float, /) -> float:
        return a / b

    print(divide(10, 2))      # ✅ صحيح
    # print(divide(a=10, b=2))
    # 💥 TypeError: a و b يجب تمريرهما بالموضع فقط
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    x = 10  # متغير عام (Global)

    def my_function():
        y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
        print(x)  # يمكن قراءة المتغير العام من داخل الدالة
        print(y)

    my_function()
    print(x)  # 10 - متاح
    # print(y)
    # 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    counter = 0

    def increment():
        counter += 1  # 💥 UnboundLocalError!

    increment()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.1: لماذا تُعتبر الدوال نقطة تحوّل في رحلتك البرمجية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-001-calculate_circle_area.py
"""

# بدون دالة - تكرار مزعج وخطير
radius1 = 5
area1 = 3.14159 * radius1 ** 2
print(f"المساحة: {area1}")

radius2 = 8
area2 = 3.14159 * radius2 ** 2
print(f"المساحة: {area2}")

# مع دالة - اكتب مرة، استخدم دائماً
def calculate_circle_area(radius: float) -> float:
    return 3.14159 * radius ** 2

print(f"المساحة: {calculate_circle_area(5)}")
print(f"المساحة: {calculate_circle_area(8)}")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-002-function_name.py
"""

def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-003-greet.py
"""

def greet(name: str) -> str:
    """تُرجع رسالة ترحيب للمستخدم."""
    return f"أهلاً بك، {name}!"

message = greet("سارة")
print(message)  # أهلاً بك، سارة!

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-004-calculate_bmi.py
"""

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-005-print_separator.py
"""

def print_separator() -> None:
    """تطبع خطاً فاصلاً. لا تُرجع أي قيمة."""
    print("-" * 40)

result = print_separator()
print(result)  # None

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-006-add.py
"""

# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-007-add.py
"""

# الصح
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)  # result = 8
print(result + 1)  # 9

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-008-greet.py
"""

def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-009-create_profile.py
"""

def create_profile(name: str, age: int, city: str) -> str:
    return f"{name}, {age} سنة، من {city}"

# موضعية (Positional) - الترتيب مهم جداً
print(create_profile("Ahmed", 25, "Riyadh"))

# مسماة (Keyword) - الترتيب لا يهم
print(create_profile(city="Riyadh", name="Ahmed", age=25))

# مزيج - الموضعية أولاً، ثم المسماة
print(create_profile("Ahmed", city="Riyadh", age=25))

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-010-ex10_3.py
"""

# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-011-ex10_3.py
"""

# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-012-greet.py
"""

def greet(name: str, greeting: str = "أهلاً") -> str:
    return f"{greeting}, {name}!"

print(greet("سارة"))              # أهلاً, سارة!
print(greet("عمر", "مرحباً"))      # مرحباً, عمر!
print(greet("لينا", greeting="صباح الخير"))  # صباح الخير, لينا!

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-013-add_item.py
"""

# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-014-add_item.py
"""

# ✅ الحل الصحيح: استخدم None كقيمة افتراضية
def add_item(item: str, cart: list | None = None) -> list:
    if cart is None:
        cart = []  # قائمة جديدة في كل استدعاء
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['banana']  ✅ صحيح الآن!

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-015-calculate_sum.py
"""

def calculate_sum(*numbers: float) -> float:
    """تجمع أي عدد من الأرقام الممررة."""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2, 3))           # 6
print(calculate_sum(1, 2, 3, 4, 5))     # 15
print(calculate_sum())                   # 0

# داخل الدالة، numbers تصبح tuple
def show_type(*args):
    print(type(args))  # <class 'tuple'>

show_type(1, 2, 3)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-016-create_user.py
"""

def create_user(**details: str) -> dict:
    """تنشئ قاموس مستخدم من أي عدد من الحقول المسماة."""
    return details

user = create_user(name="Ahmed", age=25, city="Jeddah")
print(user)
# {'name': 'Ahmed', 'age': 25, 'city': 'Jeddah'}

# داخل الدالة، kwargs تصبح dict
def show_type(**kwargs):
    print(type(kwargs))  # <class 'dict'>

show_type(a=1, b=2)

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-017-full_example.py
"""

def full_example(
    positional: str,
    default: int = 10,
    *args: int,
    keyword_only: str = "default_value",
    **kwargs: str
) -> None:
    print(f"positional: {positional}")
    print(f"default: {default}")
    print(f"args: {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"kwargs: {kwargs}")

full_example("A", 20, 1, 2, 3, keyword_only="X", extra="Y")

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-018-create_order.py
"""

# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-019-get_min_max.py
"""

def get_min_max(numbers: list[float]) -> tuple[float, float]:
    """تُرجع أصغر وأكبر قيمة في القائمة."""
    return min(numbers), max(numbers)

low, high = get_min_max([5, 2, 9, 1, 7])
print(f"الأصغر: {low}, الأكبر: {high}")  # الأصغر: 1, الأكبر: 9

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-020-calculate_discount.py
"""

def calculate_discount(price: float, is_member: bool) -> float:
    """حساب السعر بعد الخصم."""
    if price <= 0:
        return 0.0  # خروج مبكر - لا داعي لمتابعة الحساب

    if not is_member:
        return price  # خروج مبكر - لا خصم

    return price * 0.85  # الكود الرئيسي، بدون تداخل if متعدد

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-021-calculate_discount_messy.py
"""

def calculate_discount_messy(price: float, is_member: bool) -> float:
    if price > 0:
        if is_member:
            return price * 0.85
        else:
            return price
    else:
        return 0.0

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-022-log_message.py
"""

def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-023-my_function.py
"""

x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-024-increment.py
"""

counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-025-increment.py
"""

counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-026-increment.py
"""

# ✅ الطريقة الاحترافية: مرر القيمة وأعد النتيجة
def increment(counter: int) -> int:
    return counter + 1

counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)  # 2

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-027-outer.py
"""

x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-028-add_tax.py
"""

# دالة نقية (Pure Function) - يُفضّل هذا النمط
def add_tax(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# دالة غير نقية - تعتمد على وتُعدّل حالة خارجية
total = 0

def add_to_total(amount: float) -> None:
    global total
    total += amount

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-029-modify_number.py
"""

def modify_number(n: int) -> None:
    n += 10
    print(f"داخل الدالة: {n}")

x = 5
modify_number(x)
print(f"خارج الدالة: {x}")

# الناتج:
# داخل الدالة: 15
# خارج الدالة: 5   -- لم يتغير!

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-030-modify_list.py
"""

def modify_list(lst: list) -> None:
    lst.append(100)
    print(f"داخل الدالة: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"خارج الدالة: {my_list}")

# الناتج:
# داخل الدالة: [1, 2, 3, 100]
# خارج الدالة: [1, 2, 3, 100]  -- تغيّرت فعلاً!

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-031-process_safely.py
"""

def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-032-square.py
"""

def square(x: int) -> int:
    return x ** 2

# تخزين الدالة في متغير (بدون أقواس استدعاء!)
my_func = square
print(my_func(5))  # 25

# تمرير دالة كمعامل لدالة أخرى
def apply_function(func, value: int) -> int:
    return func(value)

print(apply_function(square, 6))  # 36

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.7: دوال من الدرجة الأولى (First-Class Functions) - مقدمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-033-get_score.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

def get_score(student: dict) -> int:
    return student["score"]

sorted_students = sorted(students, key=get_score, reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# الناتج:
# Sara: 92
# Ahmed: 85
# Omar: 78

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.8: Type Hints المتقدمة في الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-034-format_value.py
"""

# استخدام | للسماح بأكثر من نوع (Python 3.10+)
def format_value(value: int | float | str) -> str:
    return f"القيمة: {value}"

# استخدام Optional (أو | None) لمعامل قد يكون غائباً
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "Ahmed"}, 2: {"name": "Sara"}}
    return users.get(user_id)

# Callable: Type Hint لمعامل يكون دالة
from collections.abc import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

def add_five(x: int) -> int:
    return x + 5

print(apply_twice(add_five, 10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-035-add.py
"""

import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.9: مشروع تطبيقي: حاسبة علمية بالدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-036-log_calculation.py
"""

calculation_history: list[str] = []

def log_calculation(operation: str, result: float) -> None:
    """تسجل كل عملية حساب في سجل."""
    calculation_history.append(f"{operation} = {result}")

def show_history() -> None:
    """تعرض كل العمليات المحسوبة سابقاً."""
    if not calculation_history:
        print("لا توجد عمليات سابقة.")
        return

    print("\n📜 سجل العمليات:")
    for entry in calculation_history:
        print(f"  {entry}")

# يمكن استدعاء log_calculation بعد كل نتيجة ناجحة في run_calculator

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-037-is_even.py
"""

# تمرين 1
def is_even(number: int) -> bool:
    return number % 2 == 0

print(is_even(4))  # True
print(is_even(7))  # False

# تمرين 2
def greet_user(name: str, title: str = "الأستاذ/ة") -> str:
    return f"مرحباً {title} {name}"

print(greet_user("سمير"))                  # مرحباً الأستاذ/ة سمير
print(greet_user("نورة", title="الدكتورة"))  # مرحباً الدكتورة نورة

# تمرين 3
def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

print(average(10, 20, 30))  # 20.0

# ────────────────────────────────────────────────────────────────────────
# [076] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-038-is_valid_password.py
"""

# تمرين 4
def is_valid_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "يجب أن تكون 8 أحرف على الأقل"
    if not any(char.isdigit() for char in password):
        return False, "يجب أن تحتوي على رقم واحد على الأقل"
    if not any(char.isupper() for char in password):
        return False, "يجب أن تحتوي على حرف كبير واحد على الأقل"
    return True, "صالحة"

valid, message = is_valid_password("abc123")
print(f"{valid}: {message}")  # False: يجب أن تكون 8 أحرف على الأقل

valid, message = is_valid_password("Abcd1234")
print(f"{valid}: {message}")  # True: صالحة

# تمرين 5
def filter_products(products: list[dict], **criteria) -> list[dict]:
    result = []
    for product in products:
        if all(product.get(key) == value for key, value in criteria.items()):
            result.append(product)
    return result

products = [
    {"name": "لابتوب", "category": "إلكترونيات", "price": 3500},
    {"name": "كرسي", "category": "أثاث", "price": 450},
    {"name": "ماوس", "category": "إلكترونيات", "price": 50},
]

electronics = filter_products(products, category="إلكترونيات")
print(electronics)
# [{'name': 'لابتوب', ...}, {'name': 'ماوس', ...}]

# ────────────────────────────────────────────────────────────────────────
# [077] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-76
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-039-memoize.py
"""

def memoize(func):
    """تُرجع نسخة من func تخزّن النتائج السابقة."""
    cache: dict = {}

    def wrapper(*args):
        if args in cache:
            print(f"(من الذاكرة المؤقتة لـ {args})")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

def slow_square(n: int) -> int:
    import time
    time.sleep(1)  # محاكاة عملية بطيئة
    return n ** 2

fast_square = memoize(slow_square)

print(fast_square(5))  # يأخذ ثانية واحدة، النتيجة: 25
print(fast_square(5))  # فوري! (من الذاكرة المؤقتة لـ (5,))
print(fast_square(7))  # يأخذ ثانية واحدة، النتيجة: 49

# ────────────────────────────────────────────────────────────────────────
# [078] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-77
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.11: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-040-modify.py
"""

def modify(x, lst):
    x += 1
    lst.append(x)

a = 10
b = [1, 2, 3]
modify(a, b)
print(a, b)

# ────────────────────────────────────────────────────────────────────────
# [079] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-78
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-041-func_name.py
"""

# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل

# ────────────────────────────────────────────────────────────────────────
# [080] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-79
# ────────────────────────────────────────────────────────────────────────
def function_name(parameters) -> return_type:
    """Docstring: وصف مختصر لما تفعله الدالة."""
    # جسم الدالة
    return value

# ────────────────────────────────────────────────────────────────────────
# [081] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-80
# ────────────────────────────────────────────────────────────────────────
# خطأ شائع
def add(a: int, b: int) -> None:
    print(a + b)  # يطبع فقط، لا يُرجع!

result = add(3, 5)  # result = None وليس 8!
print(result + 1)  # 💥 TypeError: unsupported operand type(s)

# ────────────────────────────────────────────────────────────────────────
# [082] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-81
# ────────────────────────────────────────────────────────────────────────
# print(create_profile(name="Ahmed", 25, "Riyadh"))
# 💥 SyntaxError: positional argument follows keyword argument

# ────────────────────────────────────────────────────────────────────────
# [083] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-82
# ────────────────────────────────────────────────────────────────────────
# غامض - ما معنى True و False هنا؟
send_email("user@test.com", "Subject", "Body", True, False)

# واضح تماماً
send_email(
    to="user@test.com",
    subject="Subject",
    body="Body",
    is_urgent=True,
    send_copy=False
)

# ────────────────────────────────────────────────────────────────────────
# [084] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-83
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خطير جداً
def add_item(item: str, cart: list = []) -> list:
    cart.append(item)
    return cart

cart1 = add_item("apple")
print(cart1)  # ['apple']

cart2 = add_item("banana")
print(cart2)  # ['apple', 'banana']  💥 لماذا تذكرت العنصر السابق؟!

# ────────────────────────────────────────────────────────────────────────
# [085] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-84
# ────────────────────────────────────────────────────────────────────────
# استخدام * لإجبار ما بعدها أن تكون keyword-only
def create_order(product: str, *, quantity: int = 1, urgent: bool = False) -> str:
    return f"{quantity}x {product} {'(عاجل)' if urgent else ''}"

# print(create_order("Laptop", 2))
# 💥 TypeError: يجب تمرير quantity بالاسم

print(create_order("Laptop", quantity=2, urgent=True))  # ✅ صحيح

# استخدام / لإجبار ما قبلها أن تكون positional-only
def divide(a: float, b: float, /) -> float:
    return a / b

print(divide(10, 2))      # ✅ صحيح
# print(divide(a=10, b=2))
# 💥 TypeError: a و b يجب تمريرهما بالموضع فقط

# ────────────────────────────────────────────────────────────────────────
# [086] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-85
# ────────────────────────────────────────────────────────────────────────
x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!

# ────────────────────────────────────────────────────────────────────────
# [087] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-86
# ────────────────────────────────────────────────────────────────────────
counter = 0

def increment():
    counter += 1  # 💥 UnboundLocalError!

increment()

# ────────────────────────────────────────────────────────────────────────
# [088] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-87
# ────────────────────────────────────────────────────────────────────────
import math

def add(a: float, b: float) -> float:
    """جمع رقمين."""
    return a + b

def subtract(a: float, b: float) -> float:
    """طرح رقمين."""
    return a - b

def multiply(a: float, b: float) -> float:
    """ضرب رقمين."""
    return a * b

def divide(a: float, b: float) -> float | None:
    """قسمة رقمين، مع حماية من القسمة على صفر."""
    if b == 0:
        print("❌ خطأ: لا يمكن القسمة على صفر.")
        return None
    return a / b

def power(base: float, exponent: float) -> float:
    """رفع رقم لأس معين."""
    return base ** exponent

def square_root(number: float) -> float | None:
    """الجذر التربيعي، مع حماية من الأرقام السالبة."""
    if number < 0:
        print("❌ خطأ: لا يمكن حساب الجذر التربيعي لرقم سالب.")
        return None
    return math.sqrt(number)

def percentage(part: float, whole: float) -> float | None:
    """حساب النسبة المئوية لـ part من whole."""
    if whole == 0:
        print("❌ خطأ: القيمة الكلية لا يمكن أن تكون صفراً.")
        return None
    return (part / whole) * 100

# --- قاموس يربط الرمز بالدالة المقابلة (نمط احترافي) ---
OPERATIONS: dict[str, tuple] = {
    "+": (add, 2),
    "-": (subtract, 2),
    "*": (multiply, 2),
    "/": (divide, 2),
    "^": (power, 2),
    "sqrt": (square_root, 1),
    "%": (percentage, 2),
}

def display_menu() -> None:
    """طباعة قائمة العمليات المتاحة."""
    print("\n" + "=" * 40)
    print("🧮 الحاسبة العلمية")
    print("=" * 40)
    print("العمليات المتاحة:")
    print("  +     : جمع")
    print("  -     : طرح")
    print("  *     : ضرب")
    print("  /     : قسمة")
    print("  ^     : أس (قوة)")
    print("  sqrt  : جذر تربيعي")
    print("  %     : نسبة مئوية")
    print("  q     : خروج")
    print("=" * 40)

def run_calculator() -> None:
    """تشغيل الحاسبة في حلقة تفاعلية."""
    display_menu()

    while True:
        choice = input("\nاختر العملية: ").strip()

        if choice == "q":
            print("👋 إلى اللقاء!")
            break

        if choice not in OPERATIONS:
            print("❌ عملية غير معروفة. حاول مجدداً.")
            continue

        func, num_args = OPERATIONS[choice]

        try:
            if num_args == 1:
                x = float(input("أدخل الرقم: "))
                result = func(x)
            else:
                x = float(input("أدخل الرقم الأول: "))
                y = float(input("أدخل الرقم الثاني: "))
                result = func(x, y)

            if result is not None:
                print(f"✅ النتيجة: {result}")

        except ValueError:
            print("❌ خطأ: الرجاء إدخال أرقام صحيحة فقط.")

if __name__ == "__main__":
    run_calculator()

# ────────────────────────────────────────────────────────────────────────
# [089] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-10-chapter-10.ipynb#cell-88
# ────────────────────────────────────────────────────────────────────────
# تعريف دالة أساسية
def func_name(param: type) -> return_type:
    """Docstring."""
    return value

# قيم افتراضية
def func(a: int, b: int = 10) -> int: ...

# ⚠️ لا تستخدم mutable كقيمة افتراضية
def bad(lst: list = []): ...            # ❌
def good(lst: list | None = None):      # ✅
    if lst is None:
        lst = []

# *args و **kwargs
def func(*args, **kwargs): ...
func(1, 2, 3, key="value")

# Keyword-only و Positional-only
def func(a, b, /, c, d, *, e, f): ...
#         ^positional-only  ^keyword-only

# إرجاع قيم متعددة (tuple unpacking)
def minmax(nums: list) -> tuple:
    return min(nums), max(nums)
low, high = minmax([1, 5, 3])

# Scope
x = 10                   # Global
def f():
    y = 20                # Local
    global x; x = 99       # تعديل Global (تجنّبه قدر الإمكان)

# الدوال كقيم (First-Class Functions)
my_func = some_function   # بدون ()
sorted(data, key=my_func) # تمرير كمعامل
