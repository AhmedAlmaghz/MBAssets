#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — القواميس والمجموعات (Dict & Set)
Python 3.14 — From Zero to Pro — Dict

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 68 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-09-chapter-09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — القواميس والمجموعات (Dict & Set)
Python 3.14 — From Zero to Pro — Dict

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 67 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-09-chapter-09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — القواميس والمجموعات (Dict & Set)
Python 3.14 — From Zero to Pro — Dict

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 66 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    student = {"name": "Ahmed", "age": 22}

    # الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
    print(student["name"])  # Ahmed
    # print(student["grade"])  # KeyError: 'grade'  💥

    # الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
    print(student.get("grade"))        # None
    print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-09.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-09.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-09.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-09.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-09.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-09.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-09.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-09.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-09.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-09.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-09.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-09.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-09.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-09.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-09.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-09.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-09.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-09.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-09.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-09.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-09.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-09.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch09/09-001-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-001-student.py
"""

student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch09/09-002-ex9_1.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-002-ex9_1.py
"""

student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch09/09-003-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-003-ex9_2.py
"""

# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch09/09-004-person2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-004-person2.py
"""

# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch09/09-005-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-005-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch09/09-006-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-006-ex9_2.py
"""

# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch09/09-007-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-007-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch09/09-008-scores.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-008-scores.py
"""

scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch09/09-009-defaults.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-009-defaults.py
"""

# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch09/09-010-word_groups.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-010-word_groups.py
"""

from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch09/09-011-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-011-numbers.py
"""

# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch09/09-012-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-012-ex9_2.py
"""

# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch09/09-013-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-013-ex9_3.py
"""

fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch09/09-014-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-014-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch09/09-015-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-015-numbers.py
"""

numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch09/09-016-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-016-ex9_3.py
"""

# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch09/09-017-item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-017-item.py
"""

colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch09/09-018-python_devs.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-018-python_devs.py
"""

python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch09/09-019-user_a_purchases.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-019-user_a_purchases.py
"""

user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch09/09-020-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-020-ex9_3.py
"""

# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch09/09-021-big_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-021-big_list.py
"""

import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch09/09-022-allowed_ids.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-022-allowed_ids.py
"""

# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch09/09-023-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-023-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch09/09-024-frozen.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-024-frozen.py
"""

# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch09/09-025-add_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-025-add_product.py
"""

inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch09/09-026-compare_inventories.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-026-compare_inventories.py
"""

def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch09/09-027-car.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-027-car.py
"""

# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch09/09-028-team_a.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-028-team_a.py
"""

team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch09/09-029-count_words.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-029-count_words.py
"""

# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch09/09-030-orders.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-030-orders.py
"""

from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch09/09-031-ex9_7.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.7: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-031-ex9_7.py
"""

a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch09/09-032-ex9_8.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.8: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-032-ex9_8.py
"""

# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [063] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [064] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [065] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [066] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    student = {"name": "Ahmed", "age": 22}

    # الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
    print(student["name"])  # Ahmed
    # print(student["grade"])  # KeyError: 'grade'  💥

    # الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
    print(student.get("grade"))        # None
    print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-001-student.py
"""

student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-002-ex9_1.py
"""

student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-003-ex9_2.py
"""

# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-004-person2.py
"""

# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-005-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-006-ex9_2.py
"""

# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-007-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-008-scores.py
"""

scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-009-defaults.py
"""

# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-010-word_groups.py
"""

from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-011-numbers.py
"""

# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-012-ex9_2.py
"""

# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-013-ex9_3.py
"""

fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-014-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-015-numbers.py
"""

numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-016-ex9_3.py
"""

# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-017-item.py
"""

colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-018-python_devs.py
"""

python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-019-user_a_purchases.py
"""

user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-020-ex9_3.py
"""

# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-021-big_list.py
"""

import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-022-allowed_ids.py
"""

# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-023-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-024-frozen.py
"""

# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-025-add_product.py
"""

inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-026-compare_inventories.py
"""

def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-027-car.py
"""

# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-028-team_a.py
"""

team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-029-count_words.py
"""

# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-030-orders.py
"""

from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.7: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-031-ex9_7.py
"""

a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.8: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-032-ex9_8.py
"""

# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — القواميس والمجموعات (Dict & Set)
Python 3.14 — From Zero to Pro — Dict

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 66 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    student = {"name": "Ahmed", "age": 22}

    # الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
    print(student["name"])  # Ahmed
    # print(student["grade"])  # KeyError: 'grade'  💥

    # الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
    print(student.get("grade"))        # None
    print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-09.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-09.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-09.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-09.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-09.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-09.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-09.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-09.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-09.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-09.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-09.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-09.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-09.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-09.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-09.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-09.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-09.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-09.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-09.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-09.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-09.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-09.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch09/09-001-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-001-student.py
"""

student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch09/09-002-ex9_1.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-002-ex9_1.py
"""

student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch09/09-003-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-003-ex9_2.py
"""

# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch09/09-004-person2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-004-person2.py
"""

# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch09/09-005-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-005-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch09/09-006-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-006-ex9_2.py
"""

# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch09/09-007-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-007-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch09/09-008-scores.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-008-scores.py
"""

scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch09/09-009-defaults.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-009-defaults.py
"""

# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch09/09-010-word_groups.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-010-word_groups.py
"""

from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch09/09-011-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-011-numbers.py
"""

# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch09/09-012-ex9_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-012-ex9_2.py
"""

# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch09/09-013-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-013-ex9_3.py
"""

fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch09/09-014-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-014-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch09/09-015-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-015-numbers.py
"""

numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch09/09-016-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-016-ex9_3.py
"""

# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch09/09-017-item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-017-item.py
"""

colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch09/09-018-python_devs.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-018-python_devs.py
"""

python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch09/09-019-user_a_purchases.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-019-user_a_purchases.py
"""

user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch09/09-020-ex9_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-020-ex9_3.py
"""

# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch09/09-021-big_list.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-021-big_list.py
"""

import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch09/09-022-allowed_ids.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-022-allowed_ids.py
"""

# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch09/09-023-numbers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-023-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch09/09-024-frozen.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-024-frozen.py
"""

# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch09/09-025-add_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-025-add_product.py
"""

inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch09/09-026-compare_inventories.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-026-compare_inventories.py
"""

def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch09/09-027-car.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-027-car.py
"""

# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch09/09-028-team_a.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-028-team_a.py
"""

team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch09/09-029-count_words.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-029-count_words.py
"""

# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch09/09-030-orders.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-030-orders.py
"""

from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch09/09-031-ex9_7.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.7: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-031-ex9_7.py
"""

a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch09/09-032-ex9_8.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.8: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-032-ex9_8.py
"""

# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [063] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [064] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [065] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [066] dist/markdown/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    student = {"name": "Ahmed", "age": 22}

    # الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
    print(student["name"])  # Ahmed
    # print(student["grade"])  # KeyError: 'grade'  💥

    # الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
    print(student.get("grade"))        # None
    print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-001-student.py
"""

student = ["Ahmed", 22, "Computer Science", 85.5]

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.1: لماذا نحتاج هيكل بيانات جديد؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-002-ex9_1.py
"""

student: dict[str, object] = {
    "name": "Ahmed",
    "age": 22,
    "major": "Computer Science",
    "gpa": 85.5
}

print(student["name"])  # Ahmed - واضح ومباشر!

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-003-ex9_2.py
"""

# قاموس حقيقي
# "Serendipity" -> "اكتشاف شيء جميل بالصدفة"

ages: dict[str, int] = {
    "Sara": 25,
    "Omar": 30,
    "Lina": 28
}

print(ages["Sara"])  # 25

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-004-person2.py
"""

# الطريقة الأولى: الأقواس المعقوفة
empty_dict: dict = {}
person: dict[str, str] = {"name": "Khaled", "city": "Riyadh"}

# الطريقة الثانية: دالة dict()
person2 = dict(name="Noor", city="Cairo")  # بدون علامات تنصيص للمفاتيح هنا!

# الطريقة الثالثة: من قائمة أزواج (tuples)
pairs = [("a", 1), ("b", 2)]
person3 = dict(pairs)
print(person3)  # {'a': 1, 'b': 2}

# قاموس متداخل (Nested Dictionary) - شائع جداً في البيانات الحقيقية
company: dict[str, dict] = {
    "employee_1": {"name": "Sami", "salary": 8000},
    "employee_2": {"name": "Huda", "salary": 9500}
}
print(company["employee_1"]["name"])  # Sami

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-005-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-006-ex9_2.py
"""

# التحقق من وجود مفتاح قبل استخدامه
if "grade" in student:
    print(student["grade"])
else:
    print("لا توجد درجة مسجلة")

# in تبحث في المفاتيح فقط، وليس القيم!
print("name" in student)   # True
print("Ahmed" in student)  # False (هذا بحث في القيم، وهو غير مدعوم مباشرة بـ in)

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-007-student.py
"""

student = {"name": "Ahmed", "age": 22}

# الإضافة والتعديل بنفس الصيغة
student["age"] = 23           # تعديل (المفتاح موجود)
student["major"] = "AI"       # إضافة (المفتاح جديد)
print(student)
# {'name': 'Ahmed', 'age': 23, 'major': 'AI'}

# الحذف بطريقة del - تطلق خطأ إذا المفتاح غير موجود
del student["age"]

# الحذف بطريقة pop() - آمنة، وتُعيد القيمة المحذوفة
major = student.pop("major")
print(major)  # AI

# pop مع قيمة افتراضية لتجنب الخطأ
removed = student.pop("not_exist", "لا يوجد")
print(removed)  # لا يوجد

# حذف كل العناصر
student.clear()
print(student)  # {}

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-008-scores.py
"""

scores = {"Sara": 90, "Omar": 75, "Lina": 88}

# التكرار على المفاتيح فقط (الافتراضي)
for name in scores:
    print(name)

# التكرار على القيم فقط
for score in scores.values():
    print(score)

# التكرار على المفاتيح والقيم معاً - الأكثر استخداماً عملياً
for name, score in scores.items():
    print(f"{name}: {score}")

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-009-defaults.py
"""

# .update() - دمج قاموسين
defaults = {"theme": "dark", "lang": "ar", "font_size": 14}
user_settings = {"lang": "en"}

defaults.update(user_settings)
print(defaults)
# {'theme': 'dark', 'lang': 'en', 'font_size': 14}

# .setdefault() - مفيدة جداً لتجميع البيانات
word_groups: dict[str, list[str]] = {}
words = ["apple", "ant", "banana", "bee", "avocado"]

for word in words:
    first_letter = word[0]
    word_groups.setdefault(first_letter, []).append(word)

print(word_groups)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'bee']}

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-010-word_groups.py
"""

from collections import defaultdict

word_groups = defaultdict(list)
for word in words:
    word_groups[word[0]].append(word)  # لا حاجة لـ setdefault!

print(dict(word_groups))

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-011-numbers.py
"""

# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-012-ex9_2.py
"""

# d = {[1, 2]: "value"}  # TypeError: unhashable type: 'list'
d = {(1, 2): "value"}  # صحيح! tuple غير قابلة للتعديل، لذا مسموحة كمفتاح

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-013-ex9_3.py
"""

fruits: set[str] = {"apple", "banana", "apple", "orange"}
print(fruits)
# {'apple', 'banana', 'orange'}  -- لاحظ التكرار اختفى تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-014-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5] (الترتيب قد يختلف)

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-015-numbers.py
"""

numbers = [3, 1, 2, 3, 1, 4]
unique_ordered = list(dict.fromkeys(numbers))
print(unique_ordered)  # [3, 1, 2, 4] - الترتيب محفوظ!

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-016-ex9_3.py
"""

# الأقواس المعقوفة (بدون نقطتين - هذا ما يميزها عن dict)
s1: set[int] = {1, 2, 3}

# دالة set() - ضرورية للمجموعة الفارغة!
s2 = set()

# ⚠️ تحذير حرج
empty = {}        # هذا قاموس فارغ وليس مجموعة فارغة!
print(type(empty))  # <class 'dict'>

# set() من قائمة
s3 = set([1, 2, 2, 3])
print(s3)  # {1, 2, 3}

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-017-item.py
"""

colors: set[str] = {"red", "green", "blue"}

# إضافة عنصر واحد
colors.add("yellow")

# إضافة عدة عناصر
colors.update(["purple", "pink"])

# حذف عنصر - يطلق خطأ إذا غير موجود
colors.remove("red")

# حذف آمن - لا يطلق خطأ
colors.discard("not_exist")  # لا شيء يحدث

# حذف عنصر عشوائي وإرجاعه
item = colors.pop()

# التحقق من وجود عنصر - سريع جداً (انظر الأداء أدناه)
print("blue" in colors)

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-018-python_devs.py
"""

python_devs = {"Ahmed", "Sara", "Omar", "Lina"}
js_devs = {"Omar", "Lina", "Khaled", "Noor"}

# الاتحاد (Union): كل العناصر من الاثنين بدون تكرار
all_devs = python_devs | js_devs
# أو: python_devs.union(js_devs)
print(all_devs)
# {'Ahmed', 'Sara', 'Omar', 'Lina', 'Khaled', 'Noor'}

# التقاطع (Intersection): العناصر المشتركة فقط
both_devs = python_devs & js_devs
# أو: python_devs.intersection(js_devs)
print(both_devs)
# {'Omar', 'Lina'}

# الفرق (Difference): في الأولى وليس في الثانية
only_python = python_devs - js_devs
# أو: python_devs.difference(js_devs)
print(only_python)
# {'Ahmed', 'Sara'}

# الفرق التماثلي (Symmetric Difference): في أحدهما فقط وليس الاثنين
exclusive = python_devs ^ js_devs
# أو: python_devs.symmetric_difference(js_devs)
print(exclusive)
# {'Ahmed', 'Sara', 'Khaled', 'Noor'}

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-019-user_a_purchases.py
"""

user_a_purchases = {"laptop", "mouse", "keyboard"}
user_b_purchases = {"mouse", "keyboard", "monitor", "webcam"}

# ماذا اشترى B ولم يشتره A؟ (توصية محتملة لـ A)
recommendations_for_a = user_b_purchases - user_a_purchases
print(recommendations_for_a)  # {'monitor', 'webcam'}

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-020-ex9_3.py
"""

# فحص العلاقات بين المجموعات
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

print(a.issubset(b))     # True -- a مجموعة جزئية من b
print(b.issuperset(a))   # True -- b تحتوي كل عناصر a
print(a.isdisjoint({9, 10}))  # True -- لا يوجد أي عنصر مشترك

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-021-big_list.py
"""

import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-022-allowed_ids.py
"""

# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-023-numbers.py
"""

numbers = [1, 2, 2, 3, 4, 4, 5]
squared_unique = {n**2 for n in numbers}
print(squared_unique)  # {1, 4, 9, 16, 25}

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-024-frozen.py
"""

# frozenset مثل set لكنها immutable - يمكن استخدامها كمفتاح قاموس!
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # AttributeError - لا توجد طريقة للتعديل

cache: dict[frozenset, str] = {
    frozenset([1, 2]): "combo A",
    frozenset([3, 4]): "combo B"
}

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-025-add_product.py
"""

inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-026-compare_inventories.py
"""

def compare_inventories(inv_a: dict, inv_b: dict) -> dict[str, set]:
    """مقارنة منتجات مخزونين مختلفين (مثلاً فرعين لنفس المتجر)."""
    products_a = set(inv_a.keys())
    products_b = set(inv_b.keys())

    return {
        "only_in_a": products_a - products_b,
        "only_in_b": products_b - products_a,
        "in_both": products_a & products_b
    }

branch_1 = {"لابتوب": {}, "ماوس": {}, "كرسي": {}}
branch_2 = {"ماوس": {}, "كرسي": {}, "طابعة": {}}

result = compare_inventories(branch_1, branch_2)
print(result)
# {'only_in_a': {'لابتوب'}, 'only_in_b': {'طابعة'}, 'in_both': {'ماوس', 'كرسي'}}

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-027-car.py
"""

# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-028-team_a.py
"""

team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-029-count_words.py
"""

# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-030-orders.py
"""

from collections import defaultdict

orders = [
    {"product": "قلم", "quantity": 10, "price": 2.0},
    {"product": "دفتر", "quantity": 5, "price": 8.0},
    {"product": "قلم", "quantity": 7, "price": 2.0},
    {"product": "ممحاة", "quantity": 20, "price": 1.0},
    {"product": "دفتر", "quantity": 3, "price": 8.0},
]

sales_total: dict[str, float] = defaultdict(float)
quantity_total: dict[str, int] = defaultdict(int)

for order in orders:
    product = order["product"]
    sales_total[product] += order["quantity"] * order["price"]
    quantity_total[product] += order["quantity"]

print("💰 إجمالي المبيعات لكل منتج:")
for product, total in sales_total.items():
    print(f"  {product}: {total:.2f} ر.س")

best_seller = max(quantity_total, key=quantity_total.get)
print(f"\n🏆 المنتج الأكثر مبيعاً: {best_seller} ({quantity_total[best_seller]} قطعة)")

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.7: اختبر فهمك (Quiz)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-031-ex9_7.py
"""

a = {1, 2, 3}
b = {3, 4, 5}
print(a ^ b)

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.8: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-032-ex9_8.py
"""

# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
student = {"name": "Ahmed", "age": 22}

# الطريقة الأولى: الأقواس المربعة - تُطلق خطأ إذا المفتاح غير موجود
print(student["name"])  # Ahmed
# print(student["grade"])  # KeyError: 'grade'  💥

# الطريقة الثانية: get() - آمنة، تُعيد None إذا لم يوجد المفتاح
print(student.get("grade"))        # None
print(student.get("grade", "N/A"))  # N/A - قيمة افتراضية مخصصة

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
# بطيء مع قوائم كبيرة
allowed_ids = [1, 5, 10, 15, ..., 50000]  # قائمة ضخمة
for user_id in incoming_ids:
    if user_id in allowed_ids:  # بحث خطي بطيء كل مرة!
        process(user_id)

# سريع جداً
allowed_ids_set = set(allowed_ids)  # تحويل مرة واحدة
for user_id in incoming_ids:
    if user_id in allowed_ids_set:  # بحث فوري!
        process(user_id)

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-09-chapter-09.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension
