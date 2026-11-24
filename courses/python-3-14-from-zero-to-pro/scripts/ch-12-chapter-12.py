#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
Python 3.14 — From Zero to Pro — Lambda

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 88 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-12-chapter-12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
Python 3.14 — From Zero to Pro — Lambda

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-12-chapter-12 (lesson 12)
Code files: 87 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# دالة عادية
def square(x: int) -> int:
    return x ** 2

# نفس الدالة بصيغة lambda
square_lambda = lambda x: x ** 2

print(square(5))         # 25
print(square_lambda(5))  # 25

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# lambda مع عدة معاملات
add = lambda a, b: a + b
print(add(3, 4))  # 7

# lambda مع Ternary Expression
is_even = lambda n: "زوجي" if n % 2 == 0 else "فردي"
print(is_even(7))  # فردي

# lambda بدون معاملات
greet = lambda: "مرحباً!"
print(greet())  # مرحباً!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

# مع sorted() - الاستخدام الأشهر لـ lambda
sorted_by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in sorted_by_score:
    print(s["name"], s["score"])

# مع min() و max()
youngest_scorer = min(students, key=lambda s: s["score"])
print(youngest_scorer)

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(): تطبيق دالة على كل عنصر
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# filter(): تصفية العناصر حسب شرط
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# بـ map/filter
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# بـ List Comprehension - الأسلوب الـPythonic الموصى به
result = [x ** 2 for x in numbers if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# استخدام map مع دالة موجودة - مقبول تماماً وشائع
str_numbers = ["1", "2", "3"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3]

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-12.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def outer_function():
    message = "مرحباً من outer!"

    def inner_function():
        print(message)  # inner تصل لمتغير outer بفضل قاعدة LEGB (الفصل 10)

    inner_function()

outer_function()  # مرحباً من outer!

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-12.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def make_multiplier(factor: int):
    """تُرجع دالة تضرب أي رقم بـ factor."""
    def multiplier(number: int) -> int:
        return number * factor  # factor "مُغلَقة" داخل هذه الدالة
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-12.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# تأكيد عملي أن كل closure يحتفظ بنسخته الخاصة
print(double(10))  # 20 - لا تزال تتذكر factor=2
print(triple(10))  # 30 - لا تزال تتذكر factor=3

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-12.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def make_counter():
    """تُرجع دالة تعد تصاعدياً في كل استدعاء."""
    count = 0

    def increment():
        nonlocal count  # بدون هذه الكلمة، سيحدث UnboundLocalError
        count += 1
        return count

    return increment

counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # عداد جديد مستقل تماماً
print(counter2())  # 1  -- لا علاقة له بـ counter1!

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-12.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def make_validator(min_length: int):
    """تُنشئ دالة تحقق مخصصة بحد أدنى معين."""
    def validate(text: str) -> bool:
        return len(text) >= min_length
    return validate

validate_username = make_validator(5)
validate_password = make_validator(8)

print(validate_username("ali"))      # False (3 أحرف فقط)
print(validate_username("ahmed99"))  # True
print(validate_password("short"))    # False
print(validate_password("secure123"))  # True

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-12.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    """decorator بسيط يطبع رسالة قبل وبعد تنفيذ الدالة."""
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()  # استدعاء الدالة الأصلية
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

def say_hello():
    print("مرحباً!")

# التطبيق اليدوي (بدون @)
decorated_hello = my_decorator(say_hello)
decorated_hello()

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-12.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

@my_decorator  # هذا يُعادل: say_hello = my_decorator(say_hello)
def say_hello():
    print("مرحباً!")

say_hello()  # نفس النتيجة السابقة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-12.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"🔵 استدعاء {func.__name__} بالمعاملات: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"🔴 النتيجة: {result}")
        return result
    return wrapper

@my_decorator
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-12.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def bad_decorator(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
            # لا يوجد return هنا!
        return wrapper

    @bad_decorator
    def add(a, b):
        return a + b

    print(add(3, 5))  # None  💥 بدلاً من 8!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-12.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # wrapper  ⚠️ فقدنا الاسم الأصلي!
print(greet.__doc__)   # None     ⚠️ فقدنا التوثيق الأصلي!

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-12.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def my_decorator(func):
    @wraps(func)  # يحافظ على __name__ و __doc__ الأصليين
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # greet  ✅
print(greet.__doc__)   # تُرجع تحية للمستخدم.  ✅

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-12.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ {func.__name__} استغرقت {end - start:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> int:
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_calculation(1_000_000)
print(result)

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-12.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 استُدعيت: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"📝 أُرجعت: {result}")
        return result
    return wrapper

@log_calls
def divide(a: float, b: float) -> float:
    return a / b

divide(10, 2)

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-12.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from functools import wraps
    import random

    def retry(max_attempts: int = 3):
        """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        print(f"⚠️ محاولة {attempt} فشلت: {e}")
                        if attempt == max_attempts:
                            raise
                return None
            return wrapper
        return decorator

    @retry(max_attempts=3)
    def unreliable_operation():
        if random.random() < 0.7:
            raise ConnectionError("فشل الاتصال!")
        return "✅ نجحت العملية!"

    print(unreliable_operation())
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-12.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
@measure_time
@log_calls
def process(n: int) -> int:
    return n ** 2

process(10)

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-12.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property  # decorator مدمج - يجعل method يُستخدم كخاصية
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975 -- بدون أقواس استدعاء!

from functools import lru_cache

@lru_cache(maxsize=None)  # تذكر - caching تلقائي! (كتبناه يدوياً في الفصل 10)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # سريع جداً بفضل lru_cache، رغم أن fibonacci بطيئة أصلاً

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-12.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
def get_squares(n: int) -> list[int]:
    """تُنشئ قائمة كاملة بكل المربعات - تستهلك ذاكرة كبيرة!"""
    return [i ** 2 for i in range(n)]

# إذا n = 10,000,000، هذا ينشئ قائمة ضخمة بالكامل في الذاكرة فوراً!
squares = get_squares(10_000_000)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-12.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def get_squares_generator(n: int):
        """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
        for i in range(n):
            yield i ** 2  # yield بدلاً من return

    gen = get_squares_generator(5)
    print(gen)  # <generator object get_squares_generator at 0x...>

    print(next(gen))  # 0
    print(next(gen))  # 1
    print(next(gen))  # 4
    print(next(gen))  # 9
    print(next(gen))  # 16
    # print(next(gen))  # 💥 StopIteration - انتهت القيم
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-12.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# غالباً نستخدم for مباشرة بدلاً من next() اليدوية
for square in get_squares_generator(5):
    print(square)
# 0
# 1
# 4
# 9
# 16

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-12.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def demo_generator():
    print("البداية")
    yield 1
    print("بعد العنصر الأول")
    yield 2
    print("بعد العنصر الثاني")
    yield 3
    print("النهاية")

gen = demo_generator()  # لا شيء يُطبع بعد! الدالة لم تبدأ التنفيذ بعد

print("--- أول next ---")
print(next(gen))  # البداية ثم 1

print("--- ثاني next ---")
print(next(gen))  # بعد العنصر الأول ثم 2

print("--- ثالث next ---")
print(next(gen))  # بعد العنصر الثاني ثم 3

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-12.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# List Comprehension - ينشئ القائمة بالكامل فوراً في الذاكرة
squares_list = [x ** 2 for x in range(1_000_000)]

# Generator Expression - ينتج القيم عند الطلب فقط (lazy evaluation)
squares_gen = (x ** 2 for x in range(1_000_000))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-12.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
import sys

squares_list = [x ** 2 for x in range(1_000_000)]
squares_gen = (x ** 2 for x in range(1_000_000))

print(f"حجم القائمة: {sys.getsizeof(squares_list):,} بايت")
print(f"حجم المولّد: {sys.getsizeof(squares_gen):,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-12.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
gen = (x for x in range(3))

print(list(gen))  # [0, 1, 2]
print(list(gen))  # []  ⚠️ فارغة! تم استهلاك المولّد بالكامل

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-12.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def read_large_file(file_path: str):
    """تقرأ ملفاً ضخماً سطراً سطراً دون تحميله بالكامل في الذاكرة."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def count_lines_with_keyword(file_path: str, keyword: str) -> int:
    """تحسب عدد الأسطر المحتوية على كلمة معينة، بكفاءة ذاكرة عالية."""
    count = 0
    for line in read_large_file(file_path):
        if keyword in line:
            count += 1
    return count

# حتى لو كان الملف بحجم عدة جيجابايت، هذا الكود يستهلك ذاكرة ثابتة تقريباً!
# count = count_lines_with_keyword("huge_log.txt", "ERROR")

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-12.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import time
    import random
    from functools import wraps

    # --- جزء 1: Decorators للمراقبة ---

    call_log: list[dict] = []

    def monitor(func):
        """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = "نجاح"
            except Exception as e:
                result = str(e)
                status = "فشل"
                raise
            finally:
                duration = time.perf_counter() - start
                call_log.append({
                    "function": func.__name__,
                    "duration": round(duration, 6),
                    "status": status,
                })
            return result
        return wrapper

    def retry(max_attempts: int = 3, delay: float = 0.1):
        """decorator factory لإعادة المحاولة عند الفشل."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        time.sleep(delay)
                raise last_error
            return wrapper
        return decorator

    # --- جزء 2: دوال العمل الفعلية ---

    @monitor
    @retry(max_attempts=3)
    def fetch_data_from_source(source_id: int) -> dict:
        """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
        if random.random() < 0.3:
            raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
        return {"source_id": source_id, "value": random.randint(1, 100)}

    # --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

    def data_source_generator(count: int):
        """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
        for i in range(1, count + 1):
            yield i

    def process_all_sources(count: int) -> list[dict]:
        """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
        results = []
        for source_id in data_source_generator(count):
            try:
                data = fetch_data_from_source(source_id)
                results.append(data)
            except ConnectionError as e:
                print(f"❌ فشل نهائي: {e}")
        return results

    # --- جزء 4: تقرير الأداء ---

    def print_performance_report() -> None:
        print("\n" + "=" * 50)
        print("📊 تقرير مراقبة الأداء")
        print("=" * 50)

        total_calls = len(call_log)
        successful = sum(1 for log in call_log if log["status"] == "نجاح")
        total_time = sum(log["duration"] for log in call_log)

        print(f"إجمالي الاستدعاءات: {total_calls}")
        print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
        print(f"إجمالي الوقت: {total_time:.4f} ثانية")

        if call_log:
            avg_time = total_time / total_calls
            print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
        print("=" * 50)

    # --- التشغيل ---
    if __name__ == "__main__":
        results = process_all_sources(10)

        print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
        for r in results:
            print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

        print_performance_report()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [033] dist/notebooks/chapter-12.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """decorator factory يمنع استدعاء الدالة أسرع من معدل معين."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # قائمة بعنصر واحد لتجنب مشاكل nonlocal في closures متداخلة

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.perf_counter()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # بحد أقصى استدعائين في الثانية
def api_call(n: int) -> str:
    return f"نتيجة الاستدعاء {n}"

start = time.perf_counter()
for i in range(5):
    print(api_call(i))
print(f"الوقت الإجمالي: {time.perf_counter() - start:.2f} ثانية")  # ≈ 2 ثانية

# ────────────────────────────────────────────────────────────────────────
# [034] dist/notebooks/chapter-12.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
fruits = ["banana", "kiwi", "apple", "fig"]
sorted_fruits = sorted(fruits, key=lambda word: len(word))
print(sorted_fruits)  # ['fig', 'kiwi', 'apple', 'banana']

# تمرين 2
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1

# ────────────────────────────────────────────────────────────────────────
# [035] dist/notebooks/chapter-12.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
from functools import wraps

def uppercase_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_result
def get_greeting(name: str) -> str:
    return f"hello, {name}"

print(get_greeting("ahmed"))  # HELLO, AHMED

# تمرين 4
def make_bank_account(initial_balance: float):
    balance = initial_balance

    def deposit(amount: float) -> None:
        nonlocal balance
        balance += amount
        print(f"✅ تم إيداع {amount}. الرصيد الآن: {balance}")

    def check_balance() -> float:
        return balance

    return deposit, check_balance

deposit, check_balance = make_bank_account(1000)
deposit(500)                      # ✅ تم إيداع 500. الرصيد الآن: 1500
print(check_balance())            # 1500
deposit(200)                      # ✅ تم إيداع 200. الرصيد الآن: 1700
print(check_balance())            # 1700

# ────────────────────────────────────────────────────────────────────────
# [036] dist/notebooks/chapter-12.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def batch_generator(items: list, batch_size: int):
    """يقسّم قائمة إلى دفعات، مفيد عند معالجة بيانات ضخمة على أجزاء."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # إرسال أي عناصر متبقية أقل من batch_size
        yield batch

items = [1, 2, 3, 4, 5, 6, 7]
for batch in batch_generator(items, 3):
    print(batch)

# [1, 2, 3]
# [4, 5, 6]
# [7]

# ────────────────────────────────────────────────────────────────────────
# [037] dist/notebooks/chapter-12.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
def make_adder(x):
    def add(y):
        return x + y
    return add

add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch12/12-001-square.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-001-square.py
"""

# دالة عادية
def square(x: int) -> int:
    return x ** 2

# نفس الدالة بصيغة lambda
square_lambda = lambda x: x ** 2

print(square(5))         # 25
print(square_lambda(5))  # 25

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch12/12-002-add.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-002-add.py
"""

# lambda مع عدة معاملات
add = lambda a, b: a + b
print(add(3, 4))  # 7

# lambda مع Ternary Expression
is_even = lambda n: "زوجي" if n % 2 == 0 else "فردي"
print(is_even(7))  # فردي

# lambda بدون معاملات
greet = lambda: "مرحباً!"
print(greet())  # مرحباً!

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch12/12-003-square.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-003-square.py
"""

# ❌ ممارسة سيئة - استخدم def بدلاً منها
square = lambda x: x ** 2

# ✅ أفضل
def square(x: int) -> int:
    return x ** 2

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch12/12-004-students.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-004-students.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

# مع sorted() - الاستخدام الأشهر لـ lambda
sorted_by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in sorted_by_score:
    print(s["name"], s["score"])

# مع min() و max()
youngest_scorer = min(students, key=lambda s: s["score"])
print(youngest_scorer)

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch12/12-005-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-005-numbers.py
"""

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(): تطبيق دالة على كل عنصر
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# filter(): تصفية العناصر حسب شرط
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch12/12-006-result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-006-result.py
"""

# بـ map/filter
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# بـ List Comprehension - الأسلوب الـPythonic الموصى به
result = [x ** 2 for x in numbers if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch12/12-007-str_numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-007-str_numbers.py
"""

# استخدام map مع دالة موجودة - مقبول تماماً وشائع
str_numbers = ["1", "2", "3"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3]

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch12/12-008-outer_function.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-008-outer_function.py
"""

def outer_function():
    message = "مرحباً من outer!"

    def inner_function():
        print(message)  # inner تصل لمتغير outer بفضل قاعدة LEGB (الفصل 10)

    inner_function()

outer_function()  # مرحباً من outer!

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch12/12-009-make_multiplier.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-009-make_multiplier.py
"""

def make_multiplier(factor: int):
    """تُرجع دالة تضرب أي رقم بـ factor."""
    def multiplier(number: int) -> int:
        return number * factor  # factor "مُغلَقة" داخل هذه الدالة
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch12/12-010-ex12_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-010-ex12_3.py
"""

# تأكيد عملي أن كل closure يحتفظ بنسخته الخاصة
print(double(10))  # 20 - لا تزال تتذكر factor=2
print(triple(10))  # 30 - لا تزال تتذكر factor=3

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch12/12-011-make_counter.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-011-make_counter.py
"""

def make_counter():
    """تُرجع دالة تعد تصاعدياً في كل استدعاء."""
    count = 0

    def increment():
        nonlocal count  # بدون هذه الكلمة، سيحدث UnboundLocalError
        count += 1
        return count

    return increment

counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # عداد جديد مستقل تماماً
print(counter2())  # 1  -- لا علاقة له بـ counter1!

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch12/12-012-make_validator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-012-make_validator.py
"""

def make_validator(min_length: int):
    """تُنشئ دالة تحقق مخصصة بحد أدنى معين."""
    def validate(text: str) -> bool:
        return len(text) >= min_length
    return validate

validate_username = make_validator(5)
validate_password = make_validator(8)

print(validate_username("ali"))      # False (3 أحرف فقط)
print(validate_username("ahmed99"))  # True
print(validate_password("short"))    # False
print(validate_password("secure123"))  # True

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch12/12-013-my_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-013-my_decorator.py
"""

def my_decorator(func):
    """decorator بسيط يطبع رسالة قبل وبعد تنفيذ الدالة."""
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()  # استدعاء الدالة الأصلية
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

def say_hello():
    print("مرحباً!")

# التطبيق اليدوي (بدون @)
decorated_hello = my_decorator(say_hello)
decorated_hello()

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch12/12-014-my_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-014-my_decorator.py
"""

def my_decorator(func):
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

@my_decorator  # هذا يُعادل: say_hello = my_decorator(say_hello)
def say_hello():
    print("مرحباً!")

say_hello()  # نفس النتيجة السابقة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch12/12-015-my_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-015-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"🔵 استدعاء {func.__name__} بالمعاملات: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"🔴 النتيجة: {result}")
        return result
    return wrapper

@my_decorator
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch12/12-016-bad_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-016-bad_decorator.py
"""

def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
        # لا يوجد return هنا!
    return wrapper

@bad_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # None  💥 بدلاً من 8!

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch12/12-017-my_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-017-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # wrapper  ⚠️ فقدنا الاسم الأصلي!
print(greet.__doc__)   # None     ⚠️ فقدنا التوثيق الأصلي!

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch12/12-018-my_decorator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-018-my_decorator.py
"""

from functools import wraps

def my_decorator(func):
    @wraps(func)  # يحافظ على __name__ و __doc__ الأصليين
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # greet  ✅
print(greet.__doc__)   # تُرجع تحية للمستخدم.  ✅

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch12/12-019-measure_time.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-019-measure_time.py
"""

import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ {func.__name__} استغرقت {end - start:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> int:
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_calculation(1_000_000)
print(result)

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch12/12-020-log_calls.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-020-log_calls.py
"""

from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 استُدعيت: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"📝 أُرجعت: {result}")
        return result
    return wrapper

@log_calls
def divide(a: float, b: float) -> float:
    return a / b

divide(10, 2)

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch12/12-021-retry.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-021-retry.py
"""

from functools import wraps
import random

def retry(max_attempts: int = 3):
    """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ محاولة {attempt} فشلت: {e}")
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("فشل الاتصال!")
    return "✅ نجحت العملية!"

print(unreliable_operation())

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch12/12-022-require_login.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-022-require_login.py
"""

from functools import wraps

def require_login(func):
    @wraps(func)
    def wrapper(user: dict, *args, **kwargs):
        if not user.get("is_logged_in"):
            print("❌ يجب تسجيل الدخول أولاً.")
            return None
        return func(user, *args, **kwargs)
    return wrapper

@require_login
def view_profile(user: dict) -> None:
    print(f"👤 ملف {user['name']} الشخصي")

guest = {"name": "زائر", "is_logged_in": False}
logged_in_user = {"name": "أحمد", "is_logged_in": True}

view_profile(guest)          # ❌ يجب تسجيل الدخول أولاً.
view_profile(logged_in_user)  # 👤 ملف أحمد الشخصي

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch12/12-023-process.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-023-process.py
"""

@measure_time
@log_calls
def process(n: int) -> int:
    return n ** 2

process(10)

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch12/12-024-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-024-__init__.py
"""

class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property  # decorator مدمج - يجعل method يُستخدم كخاصية
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975 -- بدون أقواس استدعاء!

from functools import lru_cache

@lru_cache(maxsize=None)  # تذكر - caching تلقائي! (كتبناه يدوياً في الفصل 10)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # سريع جداً بفضل lru_cache، رغم أن fibonacci بطيئة أصلاً

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch12/12-025-get_squares.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-025-get_squares.py
"""

def get_squares(n: int) -> list[int]:
    """تُنشئ قائمة كاملة بكل المربعات - تستهلك ذاكرة كبيرة!"""
    return [i ** 2 for i in range(n)]

# إذا n = 10,000,000، هذا ينشئ قائمة ضخمة بالكامل في الذاكرة فوراً!
squares = get_squares(10_000_000)

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch12/12-026-get_squares_generator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-026-get_squares_generator.py
"""

def get_squares_generator(n: int):
    """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
    for i in range(n):
        yield i ** 2  # yield بدلاً من return

gen = get_squares_generator(5)
print(gen)  # <generator object get_squares_generator at 0x...>

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
print(next(gen))  # 9
print(next(gen))  # 16
# print(next(gen))  # 💥 StopIteration - انتهت القيم

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch12/12-027-ex12_5.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-027-ex12_5.py
"""

# غالباً نستخدم for مباشرة بدلاً من next() اليدوية
for square in get_squares_generator(5):
    print(square)
# 0
# 1
# 4
# 9
# 16

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch12/12-028-demo_generator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-028-demo_generator.py
"""

def demo_generator():
    print("البداية")
    yield 1
    print("بعد العنصر الأول")
    yield 2
    print("بعد العنصر الثاني")
    yield 3
    print("النهاية")

gen = demo_generator()  # لا شيء يُطبع بعد! الدالة لم تبدأ التنفيذ بعد

print("--- أول next ---")
print(next(gen))  # البداية ثم 1

print("--- ثاني next ---")
print(next(gen))  # بعد العنصر الأول ثم 2

print("--- ثالث next ---")
print(next(gen))  # بعد العنصر الثاني ثم 3

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch12/12-029-squares_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-029-squares_list.py
"""

# List Comprehension - ينشئ القائمة بالكامل فوراً في الذاكرة
squares_list = [x ** 2 for x in range(1_000_000)]

# Generator Expression - ينتج القيم عند الطلب فقط (lazy evaluation)
squares_gen = (x ** 2 for x in range(1_000_000))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch12/12-030-squares_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-030-squares_list.py
"""

import sys

squares_list = [x ** 2 for x in range(1_000_000)]
squares_gen = (x ** 2 for x in range(1_000_000))

print(f"حجم القائمة: {sys.getsizeof(squares_list):,} بايت")
print(f"حجم المولّد: {sys.getsizeof(squares_gen):,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch12/12-031-gen.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-031-gen.py
"""

gen = (x for x in range(3))

print(list(gen))  # [0, 1, 2]
print(list(gen))  # []  ⚠️ فارغة! تم استهلاك المولّد بالكامل

# ────────────────────────────────────────────────────────────────────────
# [069] code/chapters/ch12/12-032-read_large_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-032-read_large_file.py
"""

def read_large_file(file_path: str):
    """تقرأ ملفاً ضخماً سطراً سطراً دون تحميله بالكامل في الذاكرة."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def count_lines_with_keyword(file_path: str, keyword: str) -> int:
    """تحسب عدد الأسطر المحتوية على كلمة معينة، بكفاءة ذاكرة عالية."""
    count = 0
    for line in read_large_file(file_path):
        if keyword in line:
            count += 1
    return count

# حتى لو كان الملف بحجم عدة جيجابايت، هذا الكود يستهلك ذاكرة ثابتة تقريباً!
# count = count_lines_with_keyword("huge_log.txt", "ERROR")

# ────────────────────────────────────────────────────────────────────────
# [070] code/chapters/ch12/12-033-infinite_fibonacci.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-033-infinite_fibonacci.py
"""

def infinite_fibonacci():
    """مولّد لا نهائي لمتتالية فيبوناتشي."""
    a, b = 0, 1
    while True:  # حلقة لا نهائية - آمنة هنا لأن المولّد lazy!
        yield a
        a, b = b, a + b

fib = infinite_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# ────────────────────────────────────────────────────────────────────────
# [071] code/chapters/ch12/12-034-fib.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-034-fib.py
"""

from itertools import islice, count, cycle

# islice: أخذ جزء محدد من generator لا نهائي
fib = infinite_fibonacci()
first_5 = list(islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# count: عداد لا نهائي يبدأ من رقم معين
for i in islice(count(10, 2), 5):  # يبدأ من 10، بخطوة 2
    print(i, end=" ")
print()  # 10 12 14 16 18

# cycle: تكرار عناصر قائمة بشكل لا نهائي
colors = cycle(["red", "green", "blue"])
for color in islice(colors, 7):
    print(color, end=" ")
print()  # red green blue red green blue red

# ────────────────────────────────────────────────────────────────────────
# [072] code/chapters/ch12/12-035-monitor.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-035-monitor.py
"""

import time
import random
from functools import wraps

# --- جزء 1: Decorators للمراقبة ---

call_log: list[dict] = []

def monitor(func):
    """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            status = "نجاح"
        except Exception as e:
            result = str(e)
            status = "فشل"
            raise
        finally:
            duration = time.perf_counter() - start
            call_log.append({
                "function": func.__name__,
                "duration": round(duration, 6),
                "status": status,
            })
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 0.1):
    """decorator factory لإعادة المحاولة عند الفشل."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# --- جزء 2: دوال العمل الفعلية ---

@monitor
@retry(max_attempts=3)
def fetch_data_from_source(source_id: int) -> dict:
    """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
    if random.random() < 0.3:
        raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
    return {"source_id": source_id, "value": random.randint(1, 100)}

# --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

def data_source_generator(count: int):
    """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
    for i in range(1, count + 1):
        yield i

def process_all_sources(count: int) -> list[dict]:
    """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
    results = []
    for source_id in data_source_generator(count):
        try:
            data = fetch_data_from_source(source_id)
            results.append(data)
        except ConnectionError as e:
            print(f"❌ فشل نهائي: {e}")
    return results

# --- جزء 4: تقرير الأداء ---

def print_performance_report() -> None:
    print("\n" + "=" * 50)
    print("📊 تقرير مراقبة الأداء")
    print("=" * 50)

    total_calls = len(call_log)
    successful = sum(1 for log in call_log if log["status"] == "نجاح")
    total_time = sum(log["duration"] for log in call_log)

    print(f"إجمالي الاستدعاءات: {total_calls}")
    print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
    print(f"إجمالي الوقت: {total_time:.4f} ثانية")

    if call_log:
        avg_time = total_time / total_calls
        print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
    print("=" * 50)

# --- التشغيل ---
if __name__ == "__main__":
    results = process_all_sources(10)

    print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
    for r in results:
        print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

    print_performance_report()

# ────────────────────────────────────────────────────────────────────────
# [073] code/chapters/ch12/12-036-rate_limit.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-036-rate_limit.py
"""

import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """decorator factory يمنع استدعاء الدالة أسرع من معدل معين."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # قائمة بعنصر واحد لتجنب مشاكل nonlocal في closures متداخلة

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.perf_counter()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # بحد أقصى استدعائين في الثانية
def api_call(n: int) -> str:
    return f"نتيجة الاستدعاء {n}"

start = time.perf_counter()
for i in range(5):
    print(api_call(i))
print(f"الوقت الإجمالي: {time.perf_counter() - start:.2f} ثانية")  # ≈ 2 ثانية

# ────────────────────────────────────────────────────────────────────────
# [074] code/chapters/ch12/12-037-countdown.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-037-countdown.py
"""

# تمرين 1
fruits = ["banana", "kiwi", "apple", "fig"]
sorted_fruits = sorted(fruits, key=lambda word: len(word))
print(sorted_fruits)  # ['fig', 'kiwi', 'apple', 'banana']

# تمرين 2
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1

# ────────────────────────────────────────────────────────────────────────
# [075] code/chapters/ch12/12-038-uppercase_result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-038-uppercase_result.py
"""

# تمرين 3
from functools import wraps

def uppercase_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_result
def get_greeting(name: str) -> str:
    return f"hello, {name}"

print(get_greeting("ahmed"))  # HELLO, AHMED

# تمرين 4
def make_bank_account(initial_balance: float):
    balance = initial_balance

    def deposit(amount: float) -> None:
        nonlocal balance
        balance += amount
        print(f"✅ تم إيداع {amount}. الرصيد الآن: {balance}")

    def check_balance() -> float:
        return balance

    return deposit, check_balance

deposit, check_balance = make_bank_account(1000)
deposit(500)                      # ✅ تم إيداع 500. الرصيد الآن: 1500
print(check_balance())            # 1500
deposit(200)                      # ✅ تم إيداع 200. الرصيد الآن: 1700
print(check_balance())            # 1700

# ────────────────────────────────────────────────────────────────────────
# [076] code/chapters/ch12/12-039-batch_generator.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-039-batch_generator.py
"""

def batch_generator(items: list, batch_size: int):
    """يقسّم قائمة إلى دفعات، مفيد عند معالجة بيانات ضخمة على أجزاء."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # إرسال أي عناصر متبقية أقل من batch_size
        yield batch

items = [1, 2, 3, 4, 5, 6, 7]
for batch in batch_generator(items, 3):
    print(batch)

# [1, 2, 3]
# [4, 5, 6]
# [7]

# ────────────────────────────────────────────────────────────────────────
# [077] code/chapters/ch12/12-040-make_adder.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.9: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-040-make_adder.py
"""

def make_adder(x):
    def add(y):
        return x + y
    return add

add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))

# ────────────────────────────────────────────────────────────────────────
# [078] code/chapters/ch12/12-041-outer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-041-outer.py
"""

# ============ LAMBDA ============
square = lambda x: x ** 2              # ⚠️ تجنّب تسميتها؛ استخدم def
sorted(data, key=lambda x: x["score"]) # ✅ الاستخدام الصحيح

# ============ CLOSURE ============
def outer(x):
    def inner(y):
        return x + y   # inner "تُغلِق" على x
    return inner

add5 = outer(5)
add5(3)  # 8

# nonlocal لتعديل متغير البيئة الخارجية
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# ============ DECORATOR ============
from functools import wraps

def my_decorator(func):
    @wraps(func)               # ✅ دائماً!
    def wrapper(*args, **kwargs):
        # كود قبل
        result = func(*args, **kwargs)
        # كود بعد
        return result           # ✅ لا تنسَ return!
    return wrapper

@my_decorator
def my_function(): ...

# Decorator Factory (مع معاملات)
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(): ...

# ============ GENERATOR ============
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# Generator Expression
gen = (x ** 2 for x in range(1000000))  # lazy - كفاءة ذاكرة عالية

# itertools
from itertools import islice, count, cycle

# ────────────────────────────────────────────────────────────────────────
# [079] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ممارسة سيئة - استخدم def بدلاً منها
square = lambda x: x ** 2

# ✅ أفضل
def square(x: int) -> int:
    return x ** 2

# ────────────────────────────────────────────────────────────────────────
# [080] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
        # لا يوجد return هنا!
    return wrapper

@bad_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # None  💥 بدلاً من 8!

# ────────────────────────────────────────────────────────────────────────
# [081] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
from functools import wraps
import random

def retry(max_attempts: int = 3):
    """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ محاولة {attempt} فشلت: {e}")
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("فشل الاتصال!")
    return "✅ نجحت العملية!"

print(unreliable_operation())

# ────────────────────────────────────────────────────────────────────────
# [082] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def require_login(func):
    @wraps(func)
    def wrapper(user: dict, *args, **kwargs):
        if not user.get("is_logged_in"):
            print("❌ يجب تسجيل الدخول أولاً.")
            return None
        return func(user, *args, **kwargs)
    return wrapper

@require_login
def view_profile(user: dict) -> None:
    print(f"👤 ملف {user['name']} الشخصي")

guest = {"name": "زائر", "is_logged_in": False}
logged_in_user = {"name": "أحمد", "is_logged_in": True}

view_profile(guest)          # ❌ يجب تسجيل الدخول أولاً.
view_profile(logged_in_user)  # 👤 ملف أحمد الشخصي

# ────────────────────────────────────────────────────────────────────────
# [083] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
def get_squares_generator(n: int):
    """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
    for i in range(n):
        yield i ** 2  # yield بدلاً من return

gen = get_squares_generator(5)
print(gen)  # <generator object get_squares_generator at 0x...>

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
print(next(gen))  # 9
print(next(gen))  # 16
# print(next(gen))  # 💥 StopIteration - انتهت القيم

# ────────────────────────────────────────────────────────────────────────
# [084] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
def infinite_fibonacci():
    """مولّد لا نهائي لمتتالية فيبوناتشي."""
    a, b = 0, 1
    while True:  # حلقة لا نهائية - آمنة هنا لأن المولّد lazy!
        yield a
        a, b = b, a + b

fib = infinite_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# ────────────────────────────────────────────────────────────────────────
# [085] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
from itertools import islice, count, cycle

# islice: أخذ جزء محدد من generator لا نهائي
fib = infinite_fibonacci()
first_5 = list(islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# count: عداد لا نهائي يبدأ من رقم معين
for i in islice(count(10, 2), 5):  # يبدأ من 10، بخطوة 2
    print(i, end=" ")
print()  # 10 12 14 16 18

# cycle: تكرار عناصر قائمة بشكل لا نهائي
colors = cycle(["red", "green", "blue"])
for color in islice(colors, 7):
    print(color, end=" ")
print()  # red green blue red green blue red

# ────────────────────────────────────────────────────────────────────────
# [086] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
import time
import random
from functools import wraps

# --- جزء 1: Decorators للمراقبة ---

call_log: list[dict] = []

def monitor(func):
    """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            status = "نجاح"
        except Exception as e:
            result = str(e)
            status = "فشل"
            raise
        finally:
            duration = time.perf_counter() - start
            call_log.append({
                "function": func.__name__,
                "duration": round(duration, 6),
                "status": status,
            })
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 0.1):
    """decorator factory لإعادة المحاولة عند الفشل."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# --- جزء 2: دوال العمل الفعلية ---

@monitor
@retry(max_attempts=3)
def fetch_data_from_source(source_id: int) -> dict:
    """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
    if random.random() < 0.3:
        raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
    return {"source_id": source_id, "value": random.randint(1, 100)}

# --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

def data_source_generator(count: int):
    """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
    for i in range(1, count + 1):
        yield i

def process_all_sources(count: int) -> list[dict]:
    """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
    results = []
    for source_id in data_source_generator(count):
        try:
            data = fetch_data_from_source(source_id)
            results.append(data)
        except ConnectionError as e:
            print(f"❌ فشل نهائي: {e}")
    return results

# --- جزء 4: تقرير الأداء ---

def print_performance_report() -> None:
    print("\n" + "=" * 50)
    print("📊 تقرير مراقبة الأداء")
    print("=" * 50)

    total_calls = len(call_log)
    successful = sum(1 for log in call_log if log["status"] == "نجاح")
    total_time = sum(log["duration"] for log in call_log)

    print(f"إجمالي الاستدعاءات: {total_calls}")
    print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
    print(f"إجمالي الوقت: {total_time:.4f} ثانية")

    if call_log:
        avg_time = total_time / total_calls
        print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
    print("=" * 50)

# --- التشغيل ---
if __name__ == "__main__":
    results = process_all_sources(10)

    print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
    for r in results:
        print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

    print_performance_report()

# ────────────────────────────────────────────────────────────────────────
# [087] dist/markdown/chapter-12.md
# ────────────────────────────────────────────────────────────────────────
# ============ LAMBDA ============
square = lambda x: x ** 2              # ⚠️ تجنّب تسميتها؛ استخدم def
sorted(data, key=lambda x: x["score"]) # ✅ الاستخدام الصحيح

# ============ CLOSURE ============
def outer(x):
    def inner(y):
        return x + y   # inner "تُغلِق" على x
    return inner

add5 = outer(5)
add5(3)  # 8

# nonlocal لتعديل متغير البيئة الخارجية
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# ============ DECORATOR ============
from functools import wraps

def my_decorator(func):
    @wraps(func)               # ✅ دائماً!
    def wrapper(*args, **kwargs):
        # كود قبل
        result = func(*args, **kwargs)
        # كود بعد
        return result           # ✅ لا تنسَ return!
    return wrapper

@my_decorator
def my_function(): ...

# Decorator Factory (مع معاملات)
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(): ...

# ============ GENERATOR ============
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# Generator Expression
gen = (x ** 2 for x in range(1000000))  # lazy - كفاءة ذاكرة عالية

# itertools
from itertools import islice, count, cycle

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# دالة عادية
def square(x: int) -> int:
    return x ** 2

# نفس الدالة بصيغة lambda
square_lambda = lambda x: x ** 2

print(square(5))         # 25
print(square_lambda(5))  # 25

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# lambda مع عدة معاملات
add = lambda a, b: a + b
print(add(3, 4))  # 7

# lambda مع Ternary Expression
is_even = lambda n: "زوجي" if n % 2 == 0 else "فردي"
print(is_even(7))  # فردي

# lambda بدون معاملات
greet = lambda: "مرحباً!"
print(greet())  # مرحباً!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

# مع sorted() - الاستخدام الأشهر لـ lambda
sorted_by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in sorted_by_score:
    print(s["name"], s["score"])

# مع min() و max()
youngest_scorer = min(students, key=lambda s: s["score"])
print(youngest_scorer)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(): تطبيق دالة على كل عنصر
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# filter(): تصفية العناصر حسب شرط
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# بـ map/filter
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# بـ List Comprehension - الأسلوب الـPythonic الموصى به
result = [x ** 2 for x in numbers if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# استخدام map مع دالة موجودة - مقبول تماماً وشائع
str_numbers = ["1", "2", "3"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3]

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def outer_function():
    message = "مرحباً من outer!"

    def inner_function():
        print(message)  # inner تصل لمتغير outer بفضل قاعدة LEGB (الفصل 10)

    inner_function()

outer_function()  # مرحباً من outer!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def make_multiplier(factor: int):
    """تُرجع دالة تضرب أي رقم بـ factor."""
    def multiplier(number: int) -> int:
        return number * factor  # factor "مُغلَقة" داخل هذه الدالة
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# تأكيد عملي أن كل closure يحتفظ بنسخته الخاصة
print(double(10))  # 20 - لا تزال تتذكر factor=2
print(triple(10))  # 30 - لا تزال تتذكر factor=3

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def make_counter():
    """تُرجع دالة تعد تصاعدياً في كل استدعاء."""
    count = 0

    def increment():
        nonlocal count  # بدون هذه الكلمة، سيحدث UnboundLocalError
        count += 1
        return count

    return increment

counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # عداد جديد مستقل تماماً
print(counter2())  # 1  -- لا علاقة له بـ counter1!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def make_validator(min_length: int):
    """تُنشئ دالة تحقق مخصصة بحد أدنى معين."""
    def validate(text: str) -> bool:
        return len(text) >= min_length
    return validate

validate_username = make_validator(5)
validate_password = make_validator(8)

print(validate_username("ali"))      # False (3 أحرف فقط)
print(validate_username("ahmed99"))  # True
print(validate_password("short"))    # False
print(validate_password("secure123"))  # True

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    """decorator بسيط يطبع رسالة قبل وبعد تنفيذ الدالة."""
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()  # استدعاء الدالة الأصلية
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

def say_hello():
    print("مرحباً!")

# التطبيق اليدوي (بدون @)
decorated_hello = my_decorator(say_hello)
decorated_hello()

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

@my_decorator  # هذا يُعادل: say_hello = my_decorator(say_hello)
def say_hello():
    print("مرحباً!")

say_hello()  # نفس النتيجة السابقة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"🔵 استدعاء {func.__name__} بالمعاملات: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"🔴 النتيجة: {result}")
        return result
    return wrapper

@my_decorator
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def bad_decorator(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
            # لا يوجد return هنا!
        return wrapper

    @bad_decorator
    def add(a, b):
        return a + b

    print(add(3, 5))  # None  💥 بدلاً من 8!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # wrapper  ⚠️ فقدنا الاسم الأصلي!
print(greet.__doc__)   # None     ⚠️ فقدنا التوثيق الأصلي!

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def my_decorator(func):
    @wraps(func)  # يحافظ على __name__ و __doc__ الأصليين
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # greet  ✅
print(greet.__doc__)   # تُرجع تحية للمستخدم.  ✅

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ {func.__name__} استغرقت {end - start:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> int:
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_calculation(1_000_000)
print(result)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 استُدعيت: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"📝 أُرجعت: {result}")
        return result
    return wrapper

@log_calls
def divide(a: float, b: float) -> float:
    return a / b

divide(10, 2)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from functools import wraps
    import random

    def retry(max_attempts: int = 3):
        """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        print(f"⚠️ محاولة {attempt} فشلت: {e}")
                        if attempt == max_attempts:
                            raise
                return None
            return wrapper
        return decorator

    @retry(max_attempts=3)
    def unreliable_operation():
        if random.random() < 0.7:
            raise ConnectionError("فشل الاتصال!")
        return "✅ نجحت العملية!"

    print(unreliable_operation())
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
@measure_time
@log_calls
def process(n: int) -> int:
    return n ** 2

process(10)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property  # decorator مدمج - يجعل method يُستخدم كخاصية
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975 -- بدون أقواس استدعاء!

from functools import lru_cache

@lru_cache(maxsize=None)  # تذكر - caching تلقائي! (كتبناه يدوياً في الفصل 10)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # سريع جداً بفضل lru_cache، رغم أن fibonacci بطيئة أصلاً

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
def get_squares(n: int) -> list[int]:
    """تُنشئ قائمة كاملة بكل المربعات - تستهلك ذاكرة كبيرة!"""
    return [i ** 2 for i in range(n)]

# إذا n = 10,000,000، هذا ينشئ قائمة ضخمة بالكامل في الذاكرة فوراً!
squares = get_squares(10_000_000)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def get_squares_generator(n: int):
        """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
        for i in range(n):
            yield i ** 2  # yield بدلاً من return

    gen = get_squares_generator(5)
    print(gen)  # <generator object get_squares_generator at 0x...>

    print(next(gen))  # 0
    print(next(gen))  # 1
    print(next(gen))  # 4
    print(next(gen))  # 9
    print(next(gen))  # 16
    # print(next(gen))  # 💥 StopIteration - انتهت القيم
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# غالباً نستخدم for مباشرة بدلاً من next() اليدوية
for square in get_squares_generator(5):
    print(square)
# 0
# 1
# 4
# 9
# 16

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
def demo_generator():
    print("البداية")
    yield 1
    print("بعد العنصر الأول")
    yield 2
    print("بعد العنصر الثاني")
    yield 3
    print("النهاية")

gen = demo_generator()  # لا شيء يُطبع بعد! الدالة لم تبدأ التنفيذ بعد

print("--- أول next ---")
print(next(gen))  # البداية ثم 1

print("--- ثاني next ---")
print(next(gen))  # بعد العنصر الأول ثم 2

print("--- ثالث next ---")
print(next(gen))  # بعد العنصر الثاني ثم 3

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# List Comprehension - ينشئ القائمة بالكامل فوراً في الذاكرة
squares_list = [x ** 2 for x in range(1_000_000)]

# Generator Expression - ينتج القيم عند الطلب فقط (lazy evaluation)
squares_gen = (x ** 2 for x in range(1_000_000))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
import sys

squares_list = [x ** 2 for x in range(1_000_000)]
squares_gen = (x ** 2 for x in range(1_000_000))

print(f"حجم القائمة: {sys.getsizeof(squares_list):,} بايت")
print(f"حجم المولّد: {sys.getsizeof(squares_gen):,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
gen = (x for x in range(3))

print(list(gen))  # [0, 1, 2]
print(list(gen))  # []  ⚠️ فارغة! تم استهلاك المولّد بالكامل

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def read_large_file(file_path: str):
    """تقرأ ملفاً ضخماً سطراً سطراً دون تحميله بالكامل في الذاكرة."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def count_lines_with_keyword(file_path: str, keyword: str) -> int:
    """تحسب عدد الأسطر المحتوية على كلمة معينة، بكفاءة ذاكرة عالية."""
    count = 0
    for line in read_large_file(file_path):
        if keyword in line:
            count += 1
    return count

# حتى لو كان الملف بحجم عدة جيجابايت، هذا الكود يستهلك ذاكرة ثابتة تقريباً!
# count = count_lines_with_keyword("huge_log.txt", "ERROR")

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import time
    import random
    from functools import wraps

    # --- جزء 1: Decorators للمراقبة ---

    call_log: list[dict] = []

    def monitor(func):
        """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = "نجاح"
            except Exception as e:
                result = str(e)
                status = "فشل"
                raise
            finally:
                duration = time.perf_counter() - start
                call_log.append({
                    "function": func.__name__,
                    "duration": round(duration, 6),
                    "status": status,
                })
            return result
        return wrapper

    def retry(max_attempts: int = 3, delay: float = 0.1):
        """decorator factory لإعادة المحاولة عند الفشل."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        time.sleep(delay)
                raise last_error
            return wrapper
        return decorator

    # --- جزء 2: دوال العمل الفعلية ---

    @monitor
    @retry(max_attempts=3)
    def fetch_data_from_source(source_id: int) -> dict:
        """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
        if random.random() < 0.3:
            raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
        return {"source_id": source_id, "value": random.randint(1, 100)}

    # --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

    def data_source_generator(count: int):
        """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
        for i in range(1, count + 1):
            yield i

    def process_all_sources(count: int) -> list[dict]:
        """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
        results = []
        for source_id in data_source_generator(count):
            try:
                data = fetch_data_from_source(source_id)
                results.append(data)
            except ConnectionError as e:
                print(f"❌ فشل نهائي: {e}")
        return results

    # --- جزء 4: تقرير الأداء ---

    def print_performance_report() -> None:
        print("\n" + "=" * 50)
        print("📊 تقرير مراقبة الأداء")
        print("=" * 50)

        total_calls = len(call_log)
        successful = sum(1 for log in call_log if log["status"] == "نجاح")
        total_time = sum(log["duration"] for log in call_log)

        print(f"إجمالي الاستدعاءات: {total_calls}")
        print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
        print(f"إجمالي الوقت: {total_time:.4f} ثانية")

        if call_log:
            avg_time = total_time / total_calls
            print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
        print("=" * 50)

    # --- التشغيل ---
    if __name__ == "__main__":
        results = process_all_sources(10)

        print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
        for r in results:
            print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

        print_performance_report()
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """decorator factory يمنع استدعاء الدالة أسرع من معدل معين."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # قائمة بعنصر واحد لتجنب مشاكل nonlocal في closures متداخلة

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.perf_counter()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # بحد أقصى استدعائين في الثانية
def api_call(n: int) -> str:
    return f"نتيجة الاستدعاء {n}"

start = time.perf_counter()
for i in range(5):
    print(api_call(i))
print(f"الوقت الإجمالي: {time.perf_counter() - start:.2f} ثانية")  # ≈ 2 ثانية

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
fruits = ["banana", "kiwi", "apple", "fig"]
sorted_fruits = sorted(fruits, key=lambda word: len(word))
print(sorted_fruits)  # ['fig', 'kiwi', 'apple', 'banana']

# تمرين 2
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
from functools import wraps

def uppercase_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_result
def get_greeting(name: str) -> str:
    return f"hello, {name}"

print(get_greeting("ahmed"))  # HELLO, AHMED

# تمرين 4
def make_bank_account(initial_balance: float):
    balance = initial_balance

    def deposit(amount: float) -> None:
        nonlocal balance
        balance += amount
        print(f"✅ تم إيداع {amount}. الرصيد الآن: {balance}")

    def check_balance() -> float:
        return balance

    return deposit, check_balance

deposit, check_balance = make_bank_account(1000)
deposit(500)                      # ✅ تم إيداع 500. الرصيد الآن: 1500
print(check_balance())            # 1500
deposit(200)                      # ✅ تم إيداع 200. الرصيد الآن: 1700
print(check_balance())            # 1700

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def batch_generator(items: list, batch_size: int):
    """يقسّم قائمة إلى دفعات، مفيد عند معالجة بيانات ضخمة على أجزاء."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # إرسال أي عناصر متبقية أقل من batch_size
        yield batch

items = [1, 2, 3, 4, 5, 6, 7]
for batch in batch_generator(items, 3):
    print(batch)

# [1, 2, 3]
# [4, 5, 6]
# [7]

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
def make_adder(x):
    def add(y):
        return x + y
    return add

add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-001-square.py
"""

# دالة عادية
def square(x: int) -> int:
    return x ** 2

# نفس الدالة بصيغة lambda
square_lambda = lambda x: x ** 2

print(square(5))         # 25
print(square_lambda(5))  # 25

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-002-add.py
"""

# lambda مع عدة معاملات
add = lambda a, b: a + b
print(add(3, 4))  # 7

# lambda مع Ternary Expression
is_even = lambda n: "زوجي" if n % 2 == 0 else "فردي"
print(is_even(7))  # فردي

# lambda بدون معاملات
greet = lambda: "مرحباً!"
print(greet())  # مرحباً!

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-003-square.py
"""

# ❌ ممارسة سيئة - استخدم def بدلاً منها
square = lambda x: x ** 2

# ✅ أفضل
def square(x: int) -> int:
    return x ** 2

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-004-students.py
"""

students = [
    {"name": "Ahmed", "score": 85},
    {"name": "Sara", "score": 92},
    {"name": "Omar", "score": 78}
]

# مع sorted() - الاستخدام الأشهر لـ lambda
sorted_by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in sorted_by_score:
    print(s["name"], s["score"])

# مع min() و max()
youngest_scorer = min(students, key=lambda s: s["score"])
print(youngest_scorer)

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-005-numbers.py
"""

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(): تطبيق دالة على كل عنصر
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# filter(): تصفية العناصر حسب شرط
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-006-result.py
"""

# بـ map/filter
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# بـ List Comprehension - الأسلوب الـPythonic الموصى به
result = [x ** 2 for x in numbers if x % 2 == 0]

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.2: دوال Lambda: الدوال المجهولة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-007-str_numbers.py
"""

# استخدام map مع دالة موجودة - مقبول تماماً وشائع
str_numbers = ["1", "2", "3"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3]

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-008-outer_function.py
"""

def outer_function():
    message = "مرحباً من outer!"

    def inner_function():
        print(message)  # inner تصل لمتغير outer بفضل قاعدة LEGB (الفصل 10)

    inner_function()

outer_function()  # مرحباً من outer!

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-009-make_multiplier.py
"""

def make_multiplier(factor: int):
    """تُرجع دالة تضرب أي رقم بـ factor."""
    def multiplier(number: int) -> int:
        return number * factor  # factor "مُغلَقة" داخل هذه الدالة
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-010-ex12_3.py
"""

# تأكيد عملي أن كل closure يحتفظ بنسخته الخاصة
print(double(10))  # 20 - لا تزال تتذكر factor=2
print(triple(10))  # 30 - لا تزال تتذكر factor=3

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-011-make_counter.py
"""

def make_counter():
    """تُرجع دالة تعد تصاعدياً في كل استدعاء."""
    count = 0

    def increment():
        nonlocal count  # بدون هذه الكلمة، سيحدث UnboundLocalError
        count += 1
        return count

    return increment

counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # عداد جديد مستقل تماماً
print(counter2())  # 1  -- لا علاقة له بـ counter1!

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-012-make_validator.py
"""

def make_validator(min_length: int):
    """تُنشئ دالة تحقق مخصصة بحد أدنى معين."""
    def validate(text: str) -> bool:
        return len(text) >= min_length
    return validate

validate_username = make_validator(5)
validate_password = make_validator(8)

print(validate_username("ali"))      # False (3 أحرف فقط)
print(validate_username("ahmed99"))  # True
print(validate_password("short"))    # False
print(validate_password("secure123"))  # True

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-013-my_decorator.py
"""

def my_decorator(func):
    """decorator بسيط يطبع رسالة قبل وبعد تنفيذ الدالة."""
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()  # استدعاء الدالة الأصلية
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

def say_hello():
    print("مرحباً!")

# التطبيق اليدوي (بدون @)
decorated_hello = my_decorator(say_hello)
decorated_hello()

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-014-my_decorator.py
"""

def my_decorator(func):
    def wrapper():
        print("🔵 قبل تنفيذ الدالة...")
        func()
        print("🔴 بعد تنفيذ الدالة...")
    return wrapper

@my_decorator  # هذا يُعادل: say_hello = my_decorator(say_hello)
def say_hello():
    print("مرحباً!")

say_hello()  # نفس النتيجة السابقة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-015-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"🔵 استدعاء {func.__name__} بالمعاملات: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"🔴 النتيجة: {result}")
        return result
    return wrapper

@my_decorator
def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-016-bad_decorator.py
"""

def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
        # لا يوجد return هنا!
    return wrapper

@bad_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # None  💥 بدلاً من 8!

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-017-my_decorator.py
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # wrapper  ⚠️ فقدنا الاسم الأصلي!
print(greet.__doc__)   # None     ⚠️ فقدنا التوثيق الأصلي!

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-018-my_decorator.py
"""

from functools import wraps

def my_decorator(func):
    @wraps(func)  # يحافظ على __name__ و __doc__ الأصليين
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """تُرجع تحية للمستخدم."""
    return f"مرحباً {name}"

print(greet.__name__)  # greet  ✅
print(greet.__doc__)   # تُرجع تحية للمستخدم.  ✅

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-019-measure_time.py
"""

import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ {func.__name__} استغرقت {end - start:.4f} ثانية")
        return result
    return wrapper

@measure_time
def slow_calculation(n: int) -> int:
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_calculation(1_000_000)
print(result)

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-020-log_calls.py
"""

from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 استُدعيت: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"📝 أُرجعت: {result}")
        return result
    return wrapper

@log_calls
def divide(a: float, b: float) -> float:
    return a / b

divide(10, 2)

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-021-retry.py
"""

from functools import wraps
import random

def retry(max_attempts: int = 3):
    """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ محاولة {attempt} فشلت: {e}")
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("فشل الاتصال!")
    return "✅ نجحت العملية!"

print(unreliable_operation())

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-022-require_login.py
"""

from functools import wraps

def require_login(func):
    @wraps(func)
    def wrapper(user: dict, *args, **kwargs):
        if not user.get("is_logged_in"):
            print("❌ يجب تسجيل الدخول أولاً.")
            return None
        return func(user, *args, **kwargs)
    return wrapper

@require_login
def view_profile(user: dict) -> None:
    print(f"👤 ملف {user['name']} الشخصي")

guest = {"name": "زائر", "is_logged_in": False}
logged_in_user = {"name": "أحمد", "is_logged_in": True}

view_profile(guest)          # ❌ يجب تسجيل الدخول أولاً.
view_profile(logged_in_user)  # 👤 ملف أحمد الشخصي

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-023-process.py
"""

@measure_time
@log_calls
def process(n: int) -> int:
    return n ** 2

process(10)

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-024-__init__.py
"""

class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property  # decorator مدمج - يجعل method يُستخدم كخاصية
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975 -- بدون أقواس استدعاء!

from functools import lru_cache

@lru_cache(maxsize=None)  # تذكر - caching تلقائي! (كتبناه يدوياً في الفصل 10)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # سريع جداً بفضل lru_cache، رغم أن fibonacci بطيئة أصلاً

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-025-get_squares.py
"""

def get_squares(n: int) -> list[int]:
    """تُنشئ قائمة كاملة بكل المربعات - تستهلك ذاكرة كبيرة!"""
    return [i ** 2 for i in range(n)]

# إذا n = 10,000,000، هذا ينشئ قائمة ضخمة بالكامل في الذاكرة فوراً!
squares = get_squares(10_000_000)

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-026-get_squares_generator.py
"""

def get_squares_generator(n: int):
    """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
    for i in range(n):
        yield i ** 2  # yield بدلاً من return

gen = get_squares_generator(5)
print(gen)  # <generator object get_squares_generator at 0x...>

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
print(next(gen))  # 9
print(next(gen))  # 16
# print(next(gen))  # 💥 StopIteration - انتهت القيم

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-027-ex12_5.py
"""

# غالباً نستخدم for مباشرة بدلاً من next() اليدوية
for square in get_squares_generator(5):
    print(square)
# 0
# 1
# 4
# 9
# 16

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-028-demo_generator.py
"""

def demo_generator():
    print("البداية")
    yield 1
    print("بعد العنصر الأول")
    yield 2
    print("بعد العنصر الثاني")
    yield 3
    print("النهاية")

gen = demo_generator()  # لا شيء يُطبع بعد! الدالة لم تبدأ التنفيذ بعد

print("--- أول next ---")
print(next(gen))  # البداية ثم 1

print("--- ثاني next ---")
print(next(gen))  # بعد العنصر الأول ثم 2

print("--- ثالث next ---")
print(next(gen))  # بعد العنصر الثاني ثم 3

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-029-squares_list.py
"""

# List Comprehension - ينشئ القائمة بالكامل فوراً في الذاكرة
squares_list = [x ** 2 for x in range(1_000_000)]

# Generator Expression - ينتج القيم عند الطلب فقط (lazy evaluation)
squares_gen = (x ** 2 for x in range(1_000_000))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-030-squares_list.py
"""

import sys

squares_list = [x ** 2 for x in range(1_000_000)]
squares_gen = (x ** 2 for x in range(1_000_000))

print(f"حجم القائمة: {sys.getsizeof(squares_list):,} بايت")
print(f"حجم المولّد: {sys.getsizeof(squares_gen):,} بايت")

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-031-gen.py
"""

gen = (x for x in range(3))

print(list(gen))  # [0, 1, 2]
print(list(gen))  # []  ⚠️ فارغة! تم استهلاك المولّد بالكامل

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-032-read_large_file.py
"""

def read_large_file(file_path: str):
    """تقرأ ملفاً ضخماً سطراً سطراً دون تحميله بالكامل في الذاكرة."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def count_lines_with_keyword(file_path: str, keyword: str) -> int:
    """تحسب عدد الأسطر المحتوية على كلمة معينة، بكفاءة ذاكرة عالية."""
    count = 0
    for line in read_large_file(file_path):
        if keyword in line:
            count += 1
    return count

# حتى لو كان الملف بحجم عدة جيجابايت، هذا الكود يستهلك ذاكرة ثابتة تقريباً!
# count = count_lines_with_keyword("huge_log.txt", "ERROR")

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-033-infinite_fibonacci.py
"""

def infinite_fibonacci():
    """مولّد لا نهائي لمتتالية فيبوناتشي."""
    a, b = 0, 1
    while True:  # حلقة لا نهائية - آمنة هنا لأن المولّد lazy!
        yield a
        a, b = b, a + b

fib = infinite_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-034-fib.py
"""

from itertools import islice, count, cycle

# islice: أخذ جزء محدد من generator لا نهائي
fib = infinite_fibonacci()
first_5 = list(islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# count: عداد لا نهائي يبدأ من رقم معين
for i in islice(count(10, 2), 5):  # يبدأ من 10، بخطوة 2
    print(i, end=" ")
print()  # 10 12 14 16 18

# cycle: تكرار عناصر قائمة بشكل لا نهائي
colors = cycle(["red", "green", "blue"])
for color in islice(colors, 7):
    print(color, end=" ")
print()  # red green blue red green blue red

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-035-monitor.py
"""

import time
import random
from functools import wraps

# --- جزء 1: Decorators للمراقبة ---

call_log: list[dict] = []

def monitor(func):
    """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            status = "نجاح"
        except Exception as e:
            result = str(e)
            status = "فشل"
            raise
        finally:
            duration = time.perf_counter() - start
            call_log.append({
                "function": func.__name__,
                "duration": round(duration, 6),
                "status": status,
            })
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 0.1):
    """decorator factory لإعادة المحاولة عند الفشل."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# --- جزء 2: دوال العمل الفعلية ---

@monitor
@retry(max_attempts=3)
def fetch_data_from_source(source_id: int) -> dict:
    """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
    if random.random() < 0.3:
        raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
    return {"source_id": source_id, "value": random.randint(1, 100)}

# --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

def data_source_generator(count: int):
    """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
    for i in range(1, count + 1):
        yield i

def process_all_sources(count: int) -> list[dict]:
    """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
    results = []
    for source_id in data_source_generator(count):
        try:
            data = fetch_data_from_source(source_id)
            results.append(data)
        except ConnectionError as e:
            print(f"❌ فشل نهائي: {e}")
    return results

# --- جزء 4: تقرير الأداء ---

def print_performance_report() -> None:
    print("\n" + "=" * 50)
    print("📊 تقرير مراقبة الأداء")
    print("=" * 50)

    total_calls = len(call_log)
    successful = sum(1 for log in call_log if log["status"] == "نجاح")
    total_time = sum(log["duration"] for log in call_log)

    print(f"إجمالي الاستدعاءات: {total_calls}")
    print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
    print(f"إجمالي الوقت: {total_time:.4f} ثانية")

    if call_log:
        avg_time = total_time / total_calls
        print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
    print("=" * 50)

# --- التشغيل ---
if __name__ == "__main__":
    results = process_all_sources(10)

    print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
    for r in results:
        print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

    print_performance_report()

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-036-rate_limit.py
"""

import time
from functools import wraps

def rate_limit(calls_per_second: float):
    """decorator factory يمنع استدعاء الدالة أسرع من معدل معين."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # قائمة بعنصر واحد لتجنب مشاكل nonlocal في closures متداخلة

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.perf_counter()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)  # بحد أقصى استدعائين في الثانية
def api_call(n: int) -> str:
    return f"نتيجة الاستدعاء {n}"

start = time.perf_counter()
for i in range(5):
    print(api_call(i))
print(f"الوقت الإجمالي: {time.perf_counter() - start:.2f} ثانية")  # ≈ 2 ثانية

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-037-countdown.py
"""

# تمرين 1
fruits = ["banana", "kiwi", "apple", "fig"]
sorted_fruits = sorted(fruits, key=lambda word: len(word))
print(sorted_fruits)  # ['fig', 'kiwi', 'apple', 'banana']

# تمرين 2
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1

# ────────────────────────────────────────────────────────────────────────
# [076] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-038-uppercase_result.py
"""

# تمرين 3
from functools import wraps

def uppercase_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_result
def get_greeting(name: str) -> str:
    return f"hello, {name}"

print(get_greeting("ahmed"))  # HELLO, AHMED

# تمرين 4
def make_bank_account(initial_balance: float):
    balance = initial_balance

    def deposit(amount: float) -> None:
        nonlocal balance
        balance += amount
        print(f"✅ تم إيداع {amount}. الرصيد الآن: {balance}")

    def check_balance() -> float:
        return balance

    return deposit, check_balance

deposit, check_balance = make_bank_account(1000)
deposit(500)                      # ✅ تم إيداع 500. الرصيد الآن: 1500
print(check_balance())            # 1500
deposit(200)                      # ✅ تم إيداع 200. الرصيد الآن: 1700
print(check_balance())            # 1700

# ────────────────────────────────────────────────────────────────────────
# [077] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-76
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-039-batch_generator.py
"""

def batch_generator(items: list, batch_size: int):
    """يقسّم قائمة إلى دفعات، مفيد عند معالجة بيانات ضخمة على أجزاء."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # إرسال أي عناصر متبقية أقل من batch_size
        yield batch

items = [1, 2, 3, 4, 5, 6, 7]
for batch in batch_generator(items, 3):
    print(batch)

# [1, 2, 3]
# [4, 5, 6]
# [7]

# ────────────────────────────────────────────────────────────────────────
# [078] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-77
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.9: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-040-make_adder.py
"""

def make_adder(x):
    def add(y):
        return x + y
    return add

add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))

# ────────────────────────────────────────────────────────────────────────
# [079] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-78
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-041-outer.py
"""

# ============ LAMBDA ============
square = lambda x: x ** 2              # ⚠️ تجنّب تسميتها؛ استخدم def
sorted(data, key=lambda x: x["score"]) # ✅ الاستخدام الصحيح

# ============ CLOSURE ============
def outer(x):
    def inner(y):
        return x + y   # inner "تُغلِق" على x
    return inner

add5 = outer(5)
add5(3)  # 8

# nonlocal لتعديل متغير البيئة الخارجية
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# ============ DECORATOR ============
from functools import wraps

def my_decorator(func):
    @wraps(func)               # ✅ دائماً!
    def wrapper(*args, **kwargs):
        # كود قبل
        result = func(*args, **kwargs)
        # كود بعد
        return result           # ✅ لا تنسَ return!
    return wrapper

@my_decorator
def my_function(): ...

# Decorator Factory (مع معاملات)
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(): ...

# ============ GENERATOR ============
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# Generator Expression
gen = (x ** 2 for x in range(1000000))  # lazy - كفاءة ذاكرة عالية

# itertools
from itertools import islice, count, cycle

# ────────────────────────────────────────────────────────────────────────
# [080] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-79
# ────────────────────────────────────────────────────────────────────────
# ❌ ممارسة سيئة - استخدم def بدلاً منها
square = lambda x: x ** 2

# ✅ أفضل
def square(x: int) -> int:
    return x ** 2

# ────────────────────────────────────────────────────────────────────────
# [081] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-80
# ────────────────────────────────────────────────────────────────────────
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # ⚠️ نُفِّذت لكن النتيجة ضاعت!
        # لا يوجد return هنا!
    return wrapper

@bad_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # None  💥 بدلاً من 8!

# ────────────────────────────────────────────────────────────────────────
# [082] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-81
# ────────────────────────────────────────────────────────────────────────
from functools import wraps
import random

def retry(max_attempts: int = 3):
    """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ محاولة {attempt} فشلت: {e}")
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("فشل الاتصال!")
    return "✅ نجحت العملية!"

print(unreliable_operation())

# ────────────────────────────────────────────────────────────────────────
# [083] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-82
# ────────────────────────────────────────────────────────────────────────
from functools import wraps

def require_login(func):
    @wraps(func)
    def wrapper(user: dict, *args, **kwargs):
        if not user.get("is_logged_in"):
            print("❌ يجب تسجيل الدخول أولاً.")
            return None
        return func(user, *args, **kwargs)
    return wrapper

@require_login
def view_profile(user: dict) -> None:
    print(f"👤 ملف {user['name']} الشخصي")

guest = {"name": "زائر", "is_logged_in": False}
logged_in_user = {"name": "أحمد", "is_logged_in": True}

view_profile(guest)          # ❌ يجب تسجيل الدخول أولاً.
view_profile(logged_in_user)  # 👤 ملف أحمد الشخصي

# ────────────────────────────────────────────────────────────────────────
# [084] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-83
# ────────────────────────────────────────────────────────────────────────
def get_squares_generator(n: int):
    """تُنتج المربعات واحداً تلو الآخر، دون تخزين كل شيء في الذاكرة."""
    for i in range(n):
        yield i ** 2  # yield بدلاً من return

gen = get_squares_generator(5)
print(gen)  # <generator object get_squares_generator at 0x...>

print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
print(next(gen))  # 9
print(next(gen))  # 16
# print(next(gen))  # 💥 StopIteration - انتهت القيم

# ────────────────────────────────────────────────────────────────────────
# [085] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-84
# ────────────────────────────────────────────────────────────────────────
def infinite_fibonacci():
    """مولّد لا نهائي لمتتالية فيبوناتشي."""
    a, b = 0, 1
    while True:  # حلقة لا نهائية - آمنة هنا لأن المولّد lazy!
        yield a
        a, b = b, a + b

fib = infinite_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# ────────────────────────────────────────────────────────────────────────
# [086] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-85
# ────────────────────────────────────────────────────────────────────────
from itertools import islice, count, cycle

# islice: أخذ جزء محدد من generator لا نهائي
fib = infinite_fibonacci()
first_5 = list(islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# count: عداد لا نهائي يبدأ من رقم معين
for i in islice(count(10, 2), 5):  # يبدأ من 10، بخطوة 2
    print(i, end=" ")
print()  # 10 12 14 16 18

# cycle: تكرار عناصر قائمة بشكل لا نهائي
colors = cycle(["red", "green", "blue"])
for color in islice(colors, 7):
    print(color, end=" ")
print()  # red green blue red green blue red

# ────────────────────────────────────────────────────────────────────────
# [087] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-86
# ────────────────────────────────────────────────────────────────────────
import time
import random
from functools import wraps

# --- جزء 1: Decorators للمراقبة ---

call_log: list[dict] = []

def monitor(func):
    """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            status = "نجاح"
        except Exception as e:
            result = str(e)
            status = "فشل"
            raise
        finally:
            duration = time.perf_counter() - start
            call_log.append({
                "function": func.__name__,
                "duration": round(duration, 6),
                "status": status,
            })
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 0.1):
    """decorator factory لإعادة المحاولة عند الفشل."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# --- جزء 2: دوال العمل الفعلية ---

@monitor
@retry(max_attempts=3)
def fetch_data_from_source(source_id: int) -> dict:
    """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
    if random.random() < 0.3:
        raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
    return {"source_id": source_id, "value": random.randint(1, 100)}

# --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

def data_source_generator(count: int):
    """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
    for i in range(1, count + 1):
        yield i

def process_all_sources(count: int) -> list[dict]:
    """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
    results = []
    for source_id in data_source_generator(count):
        try:
            data = fetch_data_from_source(source_id)
            results.append(data)
        except ConnectionError as e:
            print(f"❌ فشل نهائي: {e}")
    return results

# --- جزء 4: تقرير الأداء ---

def print_performance_report() -> None:
    print("\n" + "=" * 50)
    print("📊 تقرير مراقبة الأداء")
    print("=" * 50)

    total_calls = len(call_log)
    successful = sum(1 for log in call_log if log["status"] == "نجاح")
    total_time = sum(log["duration"] for log in call_log)

    print(f"إجمالي الاستدعاءات: {total_calls}")
    print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
    print(f"إجمالي الوقت: {total_time:.4f} ثانية")

    if call_log:
        avg_time = total_time / total_calls
        print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
    print("=" * 50)

# --- التشغيل ---
if __name__ == "__main__":
    results = process_all_sources(10)

    print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
    for r in results:
        print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

    print_performance_report()

# ────────────────────────────────────────────────────────────────────────
# [088] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-12-chapter-12.ipynb#cell-87
# ────────────────────────────────────────────────────────────────────────
# ============ LAMBDA ============
square = lambda x: x ** 2              # ⚠️ تجنّب تسميتها؛ استخدم def
sorted(data, key=lambda x: x["score"]) # ✅ الاستخدام الصحيح

# ============ CLOSURE ============
def outer(x):
    def inner(y):
        return x + y   # inner "تُغلِق" على x
    return inner

add5 = outer(5)
add5(3)  # 8

# nonlocal لتعديل متغير البيئة الخارجية
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# ============ DECORATOR ============
from functools import wraps

def my_decorator(func):
    @wraps(func)               # ✅ دائماً!
    def wrapper(*args, **kwargs):
        # كود قبل
        result = func(*args, **kwargs)
        # كود بعد
        return result           # ✅ لا تنسَ return!
    return wrapper

@my_decorator
def my_function(): ...

# Decorator Factory (مع معاملات)
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(): ...

# ============ GENERATOR ============
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# Generator Expression
gen = (x ** 2 for x in range(1000000))  # lazy - كفاءة ذاكرة عالية

# itertools
from itertools import islice, count, cycle
