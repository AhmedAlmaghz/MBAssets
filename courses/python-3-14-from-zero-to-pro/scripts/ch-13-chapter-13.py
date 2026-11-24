#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة الكائنية - الأساس (Class, Object, init, self)
Python 3.14 — From Zero to Pro — Class

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 72 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-13-chapter-13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة الكائنية - الأساس (Class, Object, init, self)
Python 3.14 — From Zero to Pro — Class

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-13-chapter-13 (lesson 13)
Code files: 71 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
student1_name = "أحمد"
student1_age = 20
student1_grades = [85, 90, 78]

student2_name = "سارة"
student2_age = 22
student2_grades = [92, 88, 95]

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.grades: list[float] = []

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student1 = Student("أحمد", 20)
student1.add_grade(85)
student1.add_grade(90)
print(student1.average_grade())  # 87.5

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Dog:
    """قالب (Class) يصف أي كلب."""
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} يقول: واق واق!"

# إنشاء كائنات (Objects / Instances) من نفس القالب
dog1 = Dog("ريكس", "جيرمن شيبرد")
dog2 = Dog("لونا", "بودل")

print(dog1.bark())  # ريكس يقول: واق واق!
print(dog2.bark())  # لونا يقول: واق واق!
print(dog1.name)    # ريكس
print(dog2.breed)   # بودل

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
x = 5
print(type(x))        # <class 'int'>

name = "Ahmed"
print(type(name))     # <class 'str'>

my_list = [1, 2, 3]
print(type(my_list))  # <class 'list'>

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ClassName:
    def __init__(self, parameter1, parameter2):
        self.attribute1 = parameter1
        self.attribute2 = parameter2

    def method_name(self):
        # كود يستخدم self.attribute1 وغيرها
        pass

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class Car:
    """تمثل سيارة بخصائصها الأساسية."""

    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage: int = 0  # قيمة ابتدائية افتراضية

    def drive(self, distance: int) -> None:
        """تُحرّك السيارة مسافة معينة، وتُحدّث العداد."""
        self.mileage += distance
        print(f"🚗 قطعت {self.brand} {distance} كم. المسافة الكلية: {self.mileage} كم")

    def display_info(self) -> str:
        return f"{self.year} {self.brand} {self.model} - {self.mileage} كم"

my_car = Car("تويوتا", "كورولا", 2023)
my_car.drive(150)
my_car.drive(80)
print(my_car.display_info())

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-13.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
my_car = Car("تويوتا", "كورولا", 2023)
# هذا يُستدعي داخلياً:
# Car.__init__(my_car, "تويوتا", "كورولا", 2023)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-13.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2
print(c2.count)  # 1  -- مستقل تماماً عن c1!

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-13.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
Counter.increment(c1)

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-13.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width    # attribute
        self.height = height  # attribute

    def area(self) -> float:  # method
        return self.width * self.height

    def perimeter(self) -> float:  # method
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect.width)        # 5 - الوصول لـ attribute (بدون أقواس)
print(rect.area())       # 15.0 - استدعاء method (مع أقواس)

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-13.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class Dog:
    species = "Canis familiaris"  # Class Attribute - مشتركة بين كل الكلاب

    def __init__(self, name: str):
        self.name = name  # Instance Attribute - خاصة بكل كلب

    def __str__(self) -> str:
        return f"{self.name} ({self.species})"

dog1 = Dog("ريكس")
dog2 = Dog("لونا")

print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris -- نفس القيمة للجميع

# الوصول من الفئة مباشرة أيضاً ممكن
print(Dog.species)   # Canis familiaris

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-13.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
dog1.species = "Wolf"  # ⚠️ هذا لا يُعدّل Class Attribute!
                        # بل يُنشئ Instance Attribute جديد خاص بـ dog1 فقط!

print(dog1.species)  # Wolf (الآن Instance Attribute خاص به)
print(dog2.species)  # Canis familiaris (لم يتأثر إطلاقاً!)
print(Dog.species)   # Canis familiaris (الفئة نفسها لم تتغير)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-13.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
class Employee:
    company_name = "شركة التقنية المتقدمة"  # مشتركة للجميع
    total_employees = 0  # عداد مشترك

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1  # تعديل صحيح عبر اسم الفئة

emp1 = Employee("أحمد", 8000)
emp2 = Employee("سارة", 9500)
emp3 = Employee("عمر", 7500)

print(f"عدد الموظفين: {Employee.total_employees}")  # 3
print(f"الشركة: {emp1.company_name}")  # شركة التقنية المتقدمة

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-13.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:  # instance method
        self.balance += amount

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-13.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
class Employee:
    company_name = "شركة التقنية"
    total_employees = 0

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1

    @classmethod
    def get_total_employees(cls) -> int:
        """تستقبل cls (الفئة نفسها) بدلاً من self (كائن محدد)."""
        return cls.total_employees

    @classmethod
    def from_string(cls, data_string: str) -> "Employee":
        """Alternative Constructor - نمط شائع جداً لإنشاء كائنات من صيغ بيانات مختلفة."""
        name, salary = data_string.split(",")
        return cls(name.strip(), float(salary.strip()))

emp1 = Employee("أحمد", 8000)
emp2 = Employee.from_string("سارة, 9500")  # إنشاء من نص!

print(emp2.name, emp2.salary)              # سارة 9500.0
print(Employee.get_total_employees())      # 2

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-13.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
class MathHelper:
    @staticmethod
    def is_prime(n: int) -> bool:
        """دالة مساعدة لا تحتاج الوصول لأي بيانات خاصة بالفئة أو الكائن."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# تُستدعى عبر الفئة مباشرة، بدون إنشاء كائن!
print(MathHelper.is_prime(17))  # True
print(MathHelper.is_prime(18))  # False

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-13.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class Pizza:
    def __init__(self, toppings: list[str]):
        self.toppings = toppings

    def instance_method(self):
        return f"هذه بيتزا بإضافات: {self.toppings}"

    @classmethod
    def margherita(cls) -> "Pizza":
        return cls(["جبنة", "طماطم"])  # Alternative constructor

    @staticmethod
    def calculate_cooking_time(size_cm: int) -> int:
        return size_cm // 2  # منطق عام لا يحتاج بيانات بيتزا معينة

p1 = Pizza.margherita()
print(p1.toppings)  # ['جبنة', 'طماطم']
print(Pizza.calculate_cooking_time(30))  # 15

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-13.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # <__main__.Point object at 0x7f8a1c0b3d90>  -- غير مفيد!

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-13.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """يُستدعى عند print() أو str() - للعرض البشري المقروء."""
        return f"نقطة({self.x}, {self.y})"

    def __repr__(self) -> str:
        """يُستدعى في REPL أو داخل قوائم - للعرض التقني الدقيق."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(p)       # نقطة(3, 4)  -- يستخدم __str__
print([p])     # [Point(x=3, y=4)]  -- القوائم تستخدم __repr__ للعناصر
print(repr(p)) # Point(x=3, y=4)

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-13.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(5, 6)

print(p1 == p2)  # True  -- بدون __eq__ كانت ستكون False (مقارنة مراجع الذاكرة)!
print(p1 == p3)  # False

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-13.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(len(playlist))  # 3  -- بدون __len__ ستحصل على TypeError

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-13.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # يستدعي v1.__add__(v2) تلقائياً!
print(v3)  # Vector(4, 6)

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-13.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner           # Public - وصول حر من أي مكان
        self._balance = balance      # Protected (اصطلاحاً) - "استخدم بحذر"
        self.__pin = "1234"          # Private (اصطلاحاً) - "لا تلمس من خارج الفئة"

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-13.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # "تلميح" أنه داخلي، لكن لا شيء يمنع الوصول فعلياً

account = BankAccount(1000)
print(account._balance)  # 1000 -- يعمل! Python لا تمنعك، لكنها تُعتبر ممارسة سيئة

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-13.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class BankAccount:
        def __init__(self, pin: str):
            self.__pin = pin

        def verify_pin(self, attempt: str) -> bool:
            return self.__pin == attempt

    account = BankAccount("1234")
    print(account.verify_pin("1234"))  # True

    # print(account.__pin)
    # 💥 AttributeError: 'BankAccount' object has no attribute '__pin'
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-13.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
print(account._BankAccount__pin)  # 1234  -- يعمل تقنياً، لكن لا تفعل هذا أبداً!

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-13.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class BankAccount:
        def __init__(self, balance: float):
            self._balance = balance

        @property
        def balance(self) -> float:
            """Getter - تُستدعى عند قراءة account.balance"""
            return self._balance

        @balance.setter
        def balance(self, value: float) -> None:
            """Setter - تُستدعى عند account.balance = value"""
            if value < 0:
                raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
            self._balance = value

    account = BankAccount(1000)
    print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

    account.balance = 1500   # يستدعي الـ setter تلقائياً
    print(account.balance)   # 1500

    # account.balance = -500
    # 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-13.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
class Person:
    species = "Homo sapiens"

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self) -> None:
        print(f"مرحباً، اسمي {self.name} وعمري {self.age} سنة.")

p1 = Person("خالد", 30)
p2 = Person("منى", 25)

p1.introduce()  # مرحباً، اسمي خالد وعمري 30 سنة.
p2.introduce()  # مرحباً، اسمي منى وعمري 25 سنة.

print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-13.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"مستطيل {self.width}x{self.height}, المساحة={self.area()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return self.area() == other.area()

r1 = Rectangle(4, 5)
r2 = Rectangle(2, 10)
print(r1)              # مستطيل 4x5, المساحة=20
print(r1 == r2)         # True -- نفس المساحة رغم اختلاف الأبعاد!

# تمرين 4
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5/9

print(TemperatureConverter.celsius_to_fahrenheit(100))  # 212.0
print(TemperatureConverter.fahrenheit_to_celsius(32))   # 0.0

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-13.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
class ShoppingCart:
    def __init__(self):
        self._items: list[dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self._items.append({"name": name, "price": price, "quantity": quantity})
        print(f"✅ أُضيف {quantity}x {name} إلى السلة.")

    def total_price(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self._items)

    @property
    def item_count(self) -> int:
        """إجمالي عدد القطع (مجموع الكميات)."""
        return sum(item["quantity"] for item in self._items)

    def __len__(self) -> int:
        """عدد المنتجات المختلفة في السلة."""
        return len(self._items)

    def __str__(self) -> str:
        lines = [f"🛒 سلة التسوق ({len(self)} منتج مختلف):"]
        for item in self._items:
            subtotal = item["price"] * item["quantity"]
            lines.append(f"  - {item['name']}: {item['quantity']} × {item['price']} = {subtotal}")
        lines.append(f"الإجمالي: {self.total_price()} ر.س")
        return "\n".join(lines)

cart = ShoppingCart()
cart.add_item("لابتوب", 3500, 1)
cart.add_item("ماوس", 50, 2)
cart.add_item("كيبورد", 150, 1)

print(cart)
print(f"\nعدد القطع الكلي: {cart.item_count}")  # 4
print(f"عدد المنتجات المختلفة: {len(cart)}")    # 3

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch13/13-001-student1_name.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.1: لماذا تُعتبر البرمجة الكائنية نقلة نوعية في التفكير؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-001-student1_name.py
"""

student1_name = "أحمد"
student1_age = 20
student1_grades = [85, 90, 78]

student2_name = "سارة"
student2_age = 22
student2_grades = [92, 88, 95]

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch13/13-002-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.1: لماذا تُعتبر البرمجة الكائنية نقلة نوعية في التفكير؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-002-__init__.py
"""

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.grades: list[float] = []

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student1 = Student("أحمد", 20)
student1.add_grade(85)
student1.add_grade(90)
print(student1.average_grade())  # 87.5

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch13/13-003-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.2: المفاهيم الأساسية: Class و Object

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-003-__init__.py
"""

class Dog:
    """قالب (Class) يصف أي كلب."""
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} يقول: واق واق!"

# إنشاء كائنات (Objects / Instances) من نفس القالب
dog1 = Dog("ريكس", "جيرمن شيبرد")
dog2 = Dog("لونا", "بودل")

print(dog1.bark())  # ريكس يقول: واق واق!
print(dog2.bark())  # لونا يقول: واق واق!
print(dog1.name)    # ريكس
print(dog2.breed)   # بودل

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch13/13-004-ex13_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.2: المفاهيم الأساسية: Class و Object

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-004-ex13_2.py
"""

x = 5
print(type(x))        # <class 'int'>

name = "Ahmed"
print(type(name))     # <class 'str'>

my_list = [1, 2, 3]
print(type(my_list))  # <class 'list'>

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch13/13-005-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-005-__init__.py
"""

class ClassName:
    def __init__(self, parameter1, parameter2):
        self.attribute1 = parameter1
        self.attribute2 = parameter2

    def method_name(self):
        # كود يستخدم self.attribute1 وغيرها
        pass

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch13/13-006-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-006-__init__.py
"""

class Car:
    """تمثل سيارة بخصائصها الأساسية."""

    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage: int = 0  # قيمة ابتدائية افتراضية

    def drive(self, distance: int) -> None:
        """تُحرّك السيارة مسافة معينة، وتُحدّث العداد."""
        self.mileage += distance
        print(f"🚗 قطعت {self.brand} {distance} كم. المسافة الكلية: {self.mileage} كم")

    def display_info(self) -> str:
        return f"{self.year} {self.brand} {self.model} - {self.mileage} كم"

my_car = Car("تويوتا", "كورولا", 2023)
my_car.drive(150)
my_car.drive(80)
print(my_car.display_info())

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch13/13-007-my_car.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-007-my_car.py
"""

my_car = Car("تويوتا", "كورولا", 2023)
# هذا يُستدعي داخلياً:
# Car.__init__(my_car, "تويوتا", "كورولا", 2023)

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch13/13-008-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-008-__init__.py
"""

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2
print(c2.count)  # 1  -- مستقل تماماً عن c1!

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch13/13-009-ex13_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-009-ex13_3.py
"""

Counter.increment(c1)

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch13/13-010-increment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-010-increment.py
"""

class Counter:
    def increment():  # ❌ نسيان self!
        pass

c = Counter()
c.increment()
# 💥 TypeError: increment() takes 0 positional arguments but 1 was given

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch13/13-011-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-011-__init__.py
"""

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width    # attribute
        self.height = height  # attribute

    def area(self) -> float:  # method
        return self.width * self.height

    def perimeter(self) -> float:  # method
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect.width)        # 5 - الوصول لـ attribute (بدون أقواس)
print(rect.area())       # 15.0 - استدعاء method (مع أقواس)

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch13/13-012-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-012-__init__.py
"""

class Dog:
    species = "Canis familiaris"  # Class Attribute - مشتركة بين كل الكلاب

    def __init__(self, name: str):
        self.name = name  # Instance Attribute - خاصة بكل كلب

    def __str__(self) -> str:
        return f"{self.name} ({self.species})"

dog1 = Dog("ريكس")
dog2 = Dog("لونا")

print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris -- نفس القيمة للجميع

# الوصول من الفئة مباشرة أيضاً ممكن
print(Dog.species)   # Canis familiaris

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch13/13-013-ex13_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-013-ex13_4.py
"""

dog1.species = "Wolf"  # ⚠️ هذا لا يُعدّل Class Attribute!
                        # بل يُنشئ Instance Attribute جديد خاص بـ dog1 فقط!

print(dog1.species)  # Wolf (الآن Instance Attribute خاص به)
print(dog2.species)  # Canis familiaris (لم يتأثر إطلاقاً!)
print(Dog.species)   # Canis familiaris (الفئة نفسها لم تتغير)

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch13/13-014-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-014-__init__.py
"""

class Employee:
    company_name = "شركة التقنية المتقدمة"  # مشتركة للجميع
    total_employees = 0  # عداد مشترك

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1  # تعديل صحيح عبر اسم الفئة

emp1 = Employee("أحمد", 8000)
emp2 = Employee("سارة", 9500)
emp3 = Employee("عمر", 7500)

print(f"عدد الموظفين: {Employee.total_employees}")  # 3
print(f"الشركة: {emp1.company_name}")  # شركة التقنية المتقدمة

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch13/13-015-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-015-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:  # instance method
        self.balance += amount

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch13/13-016-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-016-__init__.py
"""

class Employee:
    company_name = "شركة التقنية"
    total_employees = 0

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1

    @classmethod
    def get_total_employees(cls) -> int:
        """تستقبل cls (الفئة نفسها) بدلاً من self (كائن محدد)."""
        return cls.total_employees

    @classmethod
    def from_string(cls, data_string: str) -> "Employee":
        """Alternative Constructor - نمط شائع جداً لإنشاء كائنات من صيغ بيانات مختلفة."""
        name, salary = data_string.split(",")
        return cls(name.strip(), float(salary.strip()))

emp1 = Employee("أحمد", 8000)
emp2 = Employee.from_string("سارة, 9500")  # إنشاء من نص!

print(emp2.name, emp2.salary)              # سارة 9500.0
print(Employee.get_total_employees())      # 2

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch13/13-017-is_prime.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-017-is_prime.py
"""

class MathHelper:
    @staticmethod
    def is_prime(n: int) -> bool:
        """دالة مساعدة لا تحتاج الوصول لأي بيانات خاصة بالفئة أو الكائن."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# تُستدعى عبر الفئة مباشرة، بدون إنشاء كائن!
print(MathHelper.is_prime(17))  # True
print(MathHelper.is_prime(18))  # False

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch13/13-018-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-018-__init__.py
"""

class Pizza:
    def __init__(self, toppings: list[str]):
        self.toppings = toppings

    def instance_method(self):
        return f"هذه بيتزا بإضافات: {self.toppings}"

    @classmethod
    def margherita(cls) -> "Pizza":
        return cls(["جبنة", "طماطم"])  # Alternative constructor

    @staticmethod
    def calculate_cooking_time(size_cm: int) -> int:
        return size_cm // 2  # منطق عام لا يحتاج بيانات بيتزا معينة

p1 = Pizza.margherita()
print(p1.toppings)  # ['جبنة', 'طماطم']
print(Pizza.calculate_cooking_time(30))  # 15

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch13/13-019-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-019-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # <__main__.Point object at 0x7f8a1c0b3d90>  -- غير مفيد!

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch13/13-020-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-020-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """يُستدعى عند print() أو str() - للعرض البشري المقروء."""
        return f"نقطة({self.x}, {self.y})"

    def __repr__(self) -> str:
        """يُستدعى في REPL أو داخل قوائم - للعرض التقني الدقيق."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(p)       # نقطة(3, 4)  -- يستخدم __str__
print([p])     # [Point(x=3, y=4)]  -- القوائم تستخدم __repr__ للعناصر
print(repr(p)) # Point(x=3, y=4)

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch13/13-021-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-021-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(5, 6)

print(p1 == p2)  # True  -- بدون __eq__ كانت ستكون False (مقارنة مراجع الذاكرة)!
print(p1 == p3)  # False

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch13/13-022-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-022-__init__.py
"""

class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(len(playlist))  # 3  -- بدون __len__ ستحصل على TypeError

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch13/13-023-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-023-__init__.py
"""

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # يستدعي v1.__add__(v2) تلقائياً!
print(v3)  # Vector(4, 6)

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch13/13-024-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-024-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner           # Public - وصول حر من أي مكان
        self._balance = balance      # Protected (اصطلاحاً) - "استخدم بحذر"
        self.__pin = "1234"          # Private (اصطلاحاً) - "لا تلمس من خارج الفئة"

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch13/13-025-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-025-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # "تلميح" أنه داخلي، لكن لا شيء يمنع الوصول فعلياً

account = BankAccount(1000)
print(account._balance)  # 1000 -- يعمل! Python لا تمنعك، لكنها تُعتبر ممارسة سيئة

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch13/13-026-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-026-__init__.py
"""

class BankAccount:
    def __init__(self, pin: str):
        self.__pin = pin

    def verify_pin(self, attempt: str) -> bool:
        return self.__pin == attempt

account = BankAccount("1234")
print(account.verify_pin("1234"))  # True

# print(account.__pin)
# 💥 AttributeError: 'BankAccount' object has no attribute '__pin'

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch13/13-027-ex13_7.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-027-ex13_7.py
"""

print(account._BankAccount__pin)  # 1234  -- يعمل تقنياً، لكن لا تفعل هذا أبداً!

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch13/13-028-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-028-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Getter - تُستدعى عند قراءة account.balance"""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Setter - تُستدعى عند account.balance = value"""
        if value < 0:
            raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
        self._balance = value

account = BankAccount(1000)
print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

account.balance = 1500   # يستدعي الـ setter تلقائياً
print(account.balance)   # 1500

# account.balance = -500
# 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch13/13-029-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-029-__init__.py
"""

class Book:
    """يمثل كتاباً واحداً في المكتبة."""

    total_books = 0  # Class Attribute - عداد لكل الكتب المُنشأة

    def __init__(self, title: str, author: str, pages: int, is_read: bool = False):
        self.title = title
        self.author = author
        self.pages = pages
        self._is_read = is_read
        Book.total_books += 1

    @property
    def is_read(self) -> bool:
        return self._is_read

    def mark_as_read(self) -> None:
        self._is_read = True
        print(f"✅ '{self.title}' تم وضع علامة 'مقروء' عليه.")

    def __str__(self) -> str:
        status = "📖 مقروء" if self._is_read else "📕 غير مقروء"
        return f"{self.title} - {self.author} ({self.pages} صفحة) [{status}]"

    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

    @classmethod
    def from_string(cls, data: str) -> "Book":
        """ينشئ كتاباً من نص بصيغة: 'العنوان|المؤلف|الصفحات'"""
        title, author, pages = data.split("|")
        return cls(title.strip(), author.strip(), int(pages.strip()))

    @staticmethod
    def estimate_reading_hours(pages: int, pages_per_hour: int = 30) -> float:
        """تقدير عدد الساعات اللازمة لقراءة كتاب بعدد صفحات معين."""
        return round(pages / pages_per_hour, 1)

class Library:
    """تمثل مكتبة تحتوي على مجموعة من الكتب."""

    def __init__(self, name: str):
        self.name = name
        self.books: list[Book] = []

    def add_book(self, book: Book) -> None:
        if book in self.books:
            print(f"⚠️ '{book.title}' موجود مسبقاً في المكتبة.")
            return
        self.books.append(book)
        print(f"✅ تمت إضافة '{book.title}' إلى مكتبة '{self.name}'.")

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                print(f"🗑️ تم حذف '{title}'.")
                return
        print(f"❌ لم يُعثر على '{title}'.")

    def get_unread_books(self) -> list[Book]:
        return [book for book in self.books if not book.is_read]

    def total_pages(self) -> int:
        return sum(book.pages for book in self.books)

    def find_by_author(self, author: str) -> list[Book]:
        return [book for book in self.books if book.author == author]

    def __len__(self) -> int:
        return len(self.books)

    def __str__(self) -> str:
        return f"مكتبة '{self.name}' - {len(self.books)} كتاب"

    def print_all(self) -> None:
        print(f"\n📚 {self}")
        print("-" * 50)
        for i, book in enumerate(self.books, start=1):
            print(f"{i}. {book}")
        print("-" * 50)

# --- التشغيل التجريبي ---
if __name__ == "__main__":
    my_library = Library("مكتبة أحمد")

    book1 = Book("نادي السيارات", "خالد عبدالله", 250)
    book2 = Book.from_string("الخيميائي | باولو كويلو | 180")
    book3 = Book("Clean Code", "Robert Martin", 400, is_read=True)

    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book1)  # محاولة إضافة مكررة

    my_library.print_all()

    book1.mark_as_read()

    print(f"\n📖 عدد الكتب غير المقروءة: {len(my_library.get_unread_books())}")
    print(f"📄 إجمالي الصفحات: {my_library.total_pages()}")
    print(f"📊 إجمالي الكتب في كل المكتبات: {Book.total_books}")

    hours = Book.estimate_reading_hours(book2.pages)
    print(f"⏱️ وقت قراءة متوقع لـ '{book2.title}': {hours} ساعة")

    print(f"\n{my_library}")  # يستخدم __str__
    print(f"عدد الكتب (len): {len(my_library)}")  # يستخدم __len__

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch13/13-030-sort_by_pages.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-030-sort_by_pages.py
"""

def sort_by_pages(self, reverse: bool = False) -> list[Book]:
    """ترتيب كتب المكتبة حسب عدد الصفحات."""
    return sorted(self.books, key=lambda book: book.pages, reverse=reverse)

# إضافة هذا كـ method داخل Library، ثم:
sorted_books = my_library.sort_by_pages()
for book in sorted_books:
    print(f"{book.title}: {book.pages} صفحة")

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch13/13-031-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-031-__init__.py
"""

class Person:
    species = "Homo sapiens"

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self) -> None:
        print(f"مرحباً، اسمي {self.name} وعمري {self.age} سنة.")

p1 = Person("خالد", 30)
p2 = Person("منى", 25)

p1.introduce()  # مرحباً، اسمي خالد وعمري 30 سنة.
p2.introduce()  # مرحباً، اسمي منى وعمري 25 سنة.

print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch13/13-032-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-032-__init__.py
"""

# تمرين 3
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"مستطيل {self.width}x{self.height}, المساحة={self.area()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return self.area() == other.area()

r1 = Rectangle(4, 5)
r2 = Rectangle(2, 10)
print(r1)              # مستطيل 4x5, المساحة=20
print(r1 == r2)         # True -- نفس المساحة رغم اختلاف الأبعاد!

# تمرين 4
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5/9

print(TemperatureConverter.celsius_to_fahrenheit(100))  # 212.0
print(TemperatureConverter.fahrenheit_to_celsius(32))   # 0.0

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch13/13-033-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-033-__init__.py
"""

class ShoppingCart:
    def __init__(self):
        self._items: list[dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self._items.append({"name": name, "price": price, "quantity": quantity})
        print(f"✅ أُضيف {quantity}x {name} إلى السلة.")

    def total_price(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self._items)

    @property
    def item_count(self) -> int:
        """إجمالي عدد القطع (مجموع الكميات)."""
        return sum(item["quantity"] for item in self._items)

    def __len__(self) -> int:
        """عدد المنتجات المختلفة في السلة."""
        return len(self._items)

    def __str__(self) -> str:
        lines = [f"🛒 سلة التسوق ({len(self)} منتج مختلف):"]
        for item in self._items:
            subtotal = item["price"] * item["quantity"]
            lines.append(f"  - {item['name']}: {item['quantity']} × {item['price']} = {subtotal}")
        lines.append(f"الإجمالي: {self.total_price()} ر.س")
        return "\n".join(lines)

cart = ShoppingCart()
cart.add_item("لابتوب", 3500, 1)
cart.add_item("ماوس", 50, 2)
cart.add_item("كيبورد", 150, 1)

print(cart)
print(f"\nعدد القطع الكلي: {cart.item_count}")  # 4
print(f"عدد المنتجات المختلفة: {len(cart)}")    # 3

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch13/13-034-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.11: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-034-__init__.py
"""

# تعريف فئة أساسية
class ClassName:
    class_attribute = "مشترك بين الجميع"  # Class Attribute

    def __init__(self, param1, param2):
        self.attr1 = param1    # Instance Attribute
        self.attr2 = param2

    def instance_method(self):          # self = الكائن نفسه
        return self.attr1

    @classmethod
    def class_method(cls):              # cls = الفئة نفسها
        return cls.class_attribute

    @staticmethod
    def static_method():                # لا self ولا cls
        return "دالة مستقلة منطقياً"

# إنشاء كائن
obj = ClassName("value1", "value2")

# Magic Methods الأساسية
class MyClass:
    def __init__(self, x): self.x = x
    def __str__(self): return f"قابل للقراءة: {self.x}"
    def __repr__(self): return f"MyClass({self.x})"
    def __eq__(self, other): return self.x == other.x
    def __len__(self): return self.x
    def __add__(self, other): return MyClass(self.x + other.x)

# خصوصية البيانات (اصطلاحية)
self.public_attr      # وصول حر
self._protected_attr  # "تحذير" - داخلي، لا تعتمد عليه من الخارج
self.__private_attr   # Name Mangling → _ClassName__private_attr

# Property - getter/setter أنيق
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("لا يمكن أن يكون سالباً")
        self._balance = value

# ────────────────────────────────────────────────────────────────────────
# [066] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
class Counter:
    def increment():  # ❌ نسيان self!
        pass

c = Counter()
c.increment()
# 💥 TypeError: increment() takes 0 positional arguments but 1 was given

# ────────────────────────────────────────────────────────────────────────
# [067] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, pin: str):
        self.__pin = pin

    def verify_pin(self, attempt: str) -> bool:
        return self.__pin == attempt

account = BankAccount("1234")
print(account.verify_pin("1234"))  # True

# print(account.__pin)
# 💥 AttributeError: 'BankAccount' object has no attribute '__pin'

# ────────────────────────────────────────────────────────────────────────
# [068] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Getter - تُستدعى عند قراءة account.balance"""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Setter - تُستدعى عند account.balance = value"""
        if value < 0:
            raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
        self._balance = value

account = BankAccount(1000)
print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

account.balance = 1500   # يستدعي الـ setter تلقائياً
print(account.balance)   # 1500

# account.balance = -500
# 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!

# ────────────────────────────────────────────────────────────────────────
# [069] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
class Book:
    """يمثل كتاباً واحداً في المكتبة."""

    total_books = 0  # Class Attribute - عداد لكل الكتب المُنشأة

    def __init__(self, title: str, author: str, pages: int, is_read: bool = False):
        self.title = title
        self.author = author
        self.pages = pages
        self._is_read = is_read
        Book.total_books += 1

    @property
    def is_read(self) -> bool:
        return self._is_read

    def mark_as_read(self) -> None:
        self._is_read = True
        print(f"✅ '{self.title}' تم وضع علامة 'مقروء' عليه.")

    def __str__(self) -> str:
        status = "📖 مقروء" if self._is_read else "📕 غير مقروء"
        return f"{self.title} - {self.author} ({self.pages} صفحة) [{status}]"

    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

    @classmethod
    def from_string(cls, data: str) -> "Book":
        """ينشئ كتاباً من نص بصيغة: 'العنوان|المؤلف|الصفحات'"""
        title, author, pages = data.split("|")
        return cls(title.strip(), author.strip(), int(pages.strip()))

    @staticmethod
    def estimate_reading_hours(pages: int, pages_per_hour: int = 30) -> float:
        """تقدير عدد الساعات اللازمة لقراءة كتاب بعدد صفحات معين."""
        return round(pages / pages_per_hour, 1)

class Library:
    """تمثل مكتبة تحتوي على مجموعة من الكتب."""

    def __init__(self, name: str):
        self.name = name
        self.books: list[Book] = []

    def add_book(self, book: Book) -> None:
        if book in self.books:
            print(f"⚠️ '{book.title}' موجود مسبقاً في المكتبة.")
            return
        self.books.append(book)
        print(f"✅ تمت إضافة '{book.title}' إلى مكتبة '{self.name}'.")

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                print(f"🗑️ تم حذف '{title}'.")
                return
        print(f"❌ لم يُعثر على '{title}'.")

    def get_unread_books(self) -> list[Book]:
        return [book for book in self.books if not book.is_read]

    def total_pages(self) -> int:
        return sum(book.pages for book in self.books)

    def find_by_author(self, author: str) -> list[Book]:
        return [book for book in self.books if book.author == author]

    def __len__(self) -> int:
        return len(self.books)

    def __str__(self) -> str:
        return f"مكتبة '{self.name}' - {len(self.books)} كتاب"

    def print_all(self) -> None:
        print(f"\n📚 {self}")
        print("-" * 50)
        for i, book in enumerate(self.books, start=1):
            print(f"{i}. {book}")
        print("-" * 50)

# --- التشغيل التجريبي ---
if __name__ == "__main__":
    my_library = Library("مكتبة أحمد")

    book1 = Book("نادي السيارات", "خالد عبدالله", 250)
    book2 = Book.from_string("الخيميائي | باولو كويلو | 180")
    book3 = Book("Clean Code", "Robert Martin", 400, is_read=True)

    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book1)  # محاولة إضافة مكررة

    my_library.print_all()

    book1.mark_as_read()

    print(f"\n📖 عدد الكتب غير المقروءة: {len(my_library.get_unread_books())}")
    print(f"📄 إجمالي الصفحات: {my_library.total_pages()}")
    print(f"📊 إجمالي الكتب في كل المكتبات: {Book.total_books}")

    hours = Book.estimate_reading_hours(book2.pages)
    print(f"⏱️ وقت قراءة متوقع لـ '{book2.title}': {hours} ساعة")

    print(f"\n{my_library}")  # يستخدم __str__
    print(f"عدد الكتب (len): {len(my_library)}")  # يستخدم __len__

# ────────────────────────────────────────────────────────────────────────
# [070] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
def sort_by_pages(self, reverse: bool = False) -> list[Book]:
    """ترتيب كتب المكتبة حسب عدد الصفحات."""
    return sorted(self.books, key=lambda book: book.pages, reverse=reverse)

# إضافة هذا كـ method داخل Library، ثم:
sorted_books = my_library.sort_by_pages()
for book in sorted_books:
    print(f"{book.title}: {book.pages} صفحة")

# ────────────────────────────────────────────────────────────────────────
# [071] dist/markdown/chapter-13.md
# ────────────────────────────────────────────────────────────────────────
# تعريف فئة أساسية
class ClassName:
    class_attribute = "مشترك بين الجميع"  # Class Attribute

    def __init__(self, param1, param2):
        self.attr1 = param1    # Instance Attribute
        self.attr2 = param2

    def instance_method(self):          # self = الكائن نفسه
        return self.attr1

    @classmethod
    def class_method(cls):              # cls = الفئة نفسها
        return cls.class_attribute

    @staticmethod
    def static_method():                # لا self ولا cls
        return "دالة مستقلة منطقياً"

# إنشاء كائن
obj = ClassName("value1", "value2")

# Magic Methods الأساسية
class MyClass:
    def __init__(self, x): self.x = x
    def __str__(self): return f"قابل للقراءة: {self.x}"
    def __repr__(self): return f"MyClass({self.x})"
    def __eq__(self, other): return self.x == other.x
    def __len__(self): return self.x
    def __add__(self, other): return MyClass(self.x + other.x)

# خصوصية البيانات (اصطلاحية)
self.public_attr      # وصول حر
self._protected_attr  # "تحذير" - داخلي، لا تعتمد عليه من الخارج
self.__private_attr   # Name Mangling → _ClassName__private_attr

# Property - getter/setter أنيق
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("لا يمكن أن يكون سالباً")
        self._balance = value

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
student1_name = "أحمد"
student1_age = 20
student1_grades = [85, 90, 78]

student2_name = "سارة"
student2_age = 22
student2_grades = [92, 88, 95]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.grades: list[float] = []

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student1 = Student("أحمد", 20)
student1.add_grade(85)
student1.add_grade(90)
print(student1.average_grade())  # 87.5

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Dog:
    """قالب (Class) يصف أي كلب."""
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} يقول: واق واق!"

# إنشاء كائنات (Objects / Instances) من نفس القالب
dog1 = Dog("ريكس", "جيرمن شيبرد")
dog2 = Dog("لونا", "بودل")

print(dog1.bark())  # ريكس يقول: واق واق!
print(dog2.bark())  # لونا يقول: واق واق!
print(dog1.name)    # ريكس
print(dog2.breed)   # بودل

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
x = 5
print(type(x))        # <class 'int'>

name = "Ahmed"
print(type(name))     # <class 'str'>

my_list = [1, 2, 3]
print(type(my_list))  # <class 'list'>

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ClassName:
    def __init__(self, parameter1, parameter2):
        self.attribute1 = parameter1
        self.attribute2 = parameter2

    def method_name(self):
        # كود يستخدم self.attribute1 وغيرها
        pass

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class Car:
    """تمثل سيارة بخصائصها الأساسية."""

    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage: int = 0  # قيمة ابتدائية افتراضية

    def drive(self, distance: int) -> None:
        """تُحرّك السيارة مسافة معينة، وتُحدّث العداد."""
        self.mileage += distance
        print(f"🚗 قطعت {self.brand} {distance} كم. المسافة الكلية: {self.mileage} كم")

    def display_info(self) -> str:
        return f"{self.year} {self.brand} {self.model} - {self.mileage} كم"

my_car = Car("تويوتا", "كورولا", 2023)
my_car.drive(150)
my_car.drive(80)
print(my_car.display_info())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
my_car = Car("تويوتا", "كورولا", 2023)
# هذا يُستدعي داخلياً:
# Car.__init__(my_car, "تويوتا", "كورولا", 2023)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2
print(c2.count)  # 1  -- مستقل تماماً عن c1!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
Counter.increment(c1)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width    # attribute
        self.height = height  # attribute

    def area(self) -> float:  # method
        return self.width * self.height

    def perimeter(self) -> float:  # method
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect.width)        # 5 - الوصول لـ attribute (بدون أقواس)
print(rect.area())       # 15.0 - استدعاء method (مع أقواس)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class Dog:
    species = "Canis familiaris"  # Class Attribute - مشتركة بين كل الكلاب

    def __init__(self, name: str):
        self.name = name  # Instance Attribute - خاصة بكل كلب

    def __str__(self) -> str:
        return f"{self.name} ({self.species})"

dog1 = Dog("ريكس")
dog2 = Dog("لونا")

print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris -- نفس القيمة للجميع

# الوصول من الفئة مباشرة أيضاً ممكن
print(Dog.species)   # Canis familiaris

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
dog1.species = "Wolf"  # ⚠️ هذا لا يُعدّل Class Attribute!
                        # بل يُنشئ Instance Attribute جديد خاص بـ dog1 فقط!

print(dog1.species)  # Wolf (الآن Instance Attribute خاص به)
print(dog2.species)  # Canis familiaris (لم يتأثر إطلاقاً!)
print(Dog.species)   # Canis familiaris (الفئة نفسها لم تتغير)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
class Employee:
    company_name = "شركة التقنية المتقدمة"  # مشتركة للجميع
    total_employees = 0  # عداد مشترك

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1  # تعديل صحيح عبر اسم الفئة

emp1 = Employee("أحمد", 8000)
emp2 = Employee("سارة", 9500)
emp3 = Employee("عمر", 7500)

print(f"عدد الموظفين: {Employee.total_employees}")  # 3
print(f"الشركة: {emp1.company_name}")  # شركة التقنية المتقدمة

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:  # instance method
        self.balance += amount

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
class Employee:
    company_name = "شركة التقنية"
    total_employees = 0

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1

    @classmethod
    def get_total_employees(cls) -> int:
        """تستقبل cls (الفئة نفسها) بدلاً من self (كائن محدد)."""
        return cls.total_employees

    @classmethod
    def from_string(cls, data_string: str) -> "Employee":
        """Alternative Constructor - نمط شائع جداً لإنشاء كائنات من صيغ بيانات مختلفة."""
        name, salary = data_string.split(",")
        return cls(name.strip(), float(salary.strip()))

emp1 = Employee("أحمد", 8000)
emp2 = Employee.from_string("سارة, 9500")  # إنشاء من نص!

print(emp2.name, emp2.salary)              # سارة 9500.0
print(Employee.get_total_employees())      # 2

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
class MathHelper:
    @staticmethod
    def is_prime(n: int) -> bool:
        """دالة مساعدة لا تحتاج الوصول لأي بيانات خاصة بالفئة أو الكائن."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# تُستدعى عبر الفئة مباشرة، بدون إنشاء كائن!
print(MathHelper.is_prime(17))  # True
print(MathHelper.is_prime(18))  # False

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class Pizza:
    def __init__(self, toppings: list[str]):
        self.toppings = toppings

    def instance_method(self):
        return f"هذه بيتزا بإضافات: {self.toppings}"

    @classmethod
    def margherita(cls) -> "Pizza":
        return cls(["جبنة", "طماطم"])  # Alternative constructor

    @staticmethod
    def calculate_cooking_time(size_cm: int) -> int:
        return size_cm // 2  # منطق عام لا يحتاج بيانات بيتزا معينة

p1 = Pizza.margherita()
print(p1.toppings)  # ['جبنة', 'طماطم']
print(Pizza.calculate_cooking_time(30))  # 15

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # <__main__.Point object at 0x7f8a1c0b3d90>  -- غير مفيد!

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """يُستدعى عند print() أو str() - للعرض البشري المقروء."""
        return f"نقطة({self.x}, {self.y})"

    def __repr__(self) -> str:
        """يُستدعى في REPL أو داخل قوائم - للعرض التقني الدقيق."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(p)       # نقطة(3, 4)  -- يستخدم __str__
print([p])     # [Point(x=3, y=4)]  -- القوائم تستخدم __repr__ للعناصر
print(repr(p)) # Point(x=3, y=4)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(5, 6)

print(p1 == p2)  # True  -- بدون __eq__ كانت ستكون False (مقارنة مراجع الذاكرة)!
print(p1 == p3)  # False

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(len(playlist))  # 3  -- بدون __len__ ستحصل على TypeError

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # يستدعي v1.__add__(v2) تلقائياً!
print(v3)  # Vector(4, 6)

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner           # Public - وصول حر من أي مكان
        self._balance = balance      # Protected (اصطلاحاً) - "استخدم بحذر"
        self.__pin = "1234"          # Private (اصطلاحاً) - "لا تلمس من خارج الفئة"

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # "تلميح" أنه داخلي، لكن لا شيء يمنع الوصول فعلياً

account = BankAccount(1000)
print(account._balance)  # 1000 -- يعمل! Python لا تمنعك، لكنها تُعتبر ممارسة سيئة

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class BankAccount:
        def __init__(self, pin: str):
            self.__pin = pin

        def verify_pin(self, attempt: str) -> bool:
            return self.__pin == attempt

    account = BankAccount("1234")
    print(account.verify_pin("1234"))  # True

    # print(account.__pin)
    # 💥 AttributeError: 'BankAccount' object has no attribute '__pin'
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
print(account._BankAccount__pin)  # 1234  -- يعمل تقنياً، لكن لا تفعل هذا أبداً!

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class BankAccount:
        def __init__(self, balance: float):
            self._balance = balance

        @property
        def balance(self) -> float:
            """Getter - تُستدعى عند قراءة account.balance"""
            return self._balance

        @balance.setter
        def balance(self, value: float) -> None:
            """Setter - تُستدعى عند account.balance = value"""
            if value < 0:
                raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
            self._balance = value

    account = BankAccount(1000)
    print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

    account.balance = 1500   # يستدعي الـ setter تلقائياً
    print(account.balance)   # 1500

    # account.balance = -500
    # 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
class Person:
    species = "Homo sapiens"

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self) -> None:
        print(f"مرحباً، اسمي {self.name} وعمري {self.age} سنة.")

p1 = Person("خالد", 30)
p2 = Person("منى", 25)

p1.introduce()  # مرحباً، اسمي خالد وعمري 30 سنة.
p2.introduce()  # مرحباً، اسمي منى وعمري 25 سنة.

print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"مستطيل {self.width}x{self.height}, المساحة={self.area()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return self.area() == other.area()

r1 = Rectangle(4, 5)
r2 = Rectangle(2, 10)
print(r1)              # مستطيل 4x5, المساحة=20
print(r1 == r2)         # True -- نفس المساحة رغم اختلاف الأبعاد!

# تمرين 4
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5/9

print(TemperatureConverter.celsius_to_fahrenheit(100))  # 212.0
print(TemperatureConverter.fahrenheit_to_celsius(32))   # 0.0

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
class ShoppingCart:
    def __init__(self):
        self._items: list[dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self._items.append({"name": name, "price": price, "quantity": quantity})
        print(f"✅ أُضيف {quantity}x {name} إلى السلة.")

    def total_price(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self._items)

    @property
    def item_count(self) -> int:
        """إجمالي عدد القطع (مجموع الكميات)."""
        return sum(item["quantity"] for item in self._items)

    def __len__(self) -> int:
        """عدد المنتجات المختلفة في السلة."""
        return len(self._items)

    def __str__(self) -> str:
        lines = [f"🛒 سلة التسوق ({len(self)} منتج مختلف):"]
        for item in self._items:
            subtotal = item["price"] * item["quantity"]
            lines.append(f"  - {item['name']}: {item['quantity']} × {item['price']} = {subtotal}")
        lines.append(f"الإجمالي: {self.total_price()} ر.س")
        return "\n".join(lines)

cart = ShoppingCart()
cart.add_item("لابتوب", 3500, 1)
cart.add_item("ماوس", 50, 2)
cart.add_item("كيبورد", 150, 1)

print(cart)
print(f"\nعدد القطع الكلي: {cart.item_count}")  # 4
print(f"عدد المنتجات المختلفة: {len(cart)}")    # 3

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.1: لماذا تُعتبر البرمجة الكائنية نقلة نوعية في التفكير؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-001-student1_name.py
"""

student1_name = "أحمد"
student1_age = 20
student1_grades = [85, 90, 78]

student2_name = "سارة"
student2_age = 22
student2_grades = [92, 88, 95]

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.1: لماذا تُعتبر البرمجة الكائنية نقلة نوعية في التفكير؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-002-__init__.py
"""

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.grades: list[float] = []

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student1 = Student("أحمد", 20)
student1.add_grade(85)
student1.add_grade(90)
print(student1.average_grade())  # 87.5

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.2: المفاهيم الأساسية: Class و Object

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-003-__init__.py
"""

class Dog:
    """قالب (Class) يصف أي كلب."""
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} يقول: واق واق!"

# إنشاء كائنات (Objects / Instances) من نفس القالب
dog1 = Dog("ريكس", "جيرمن شيبرد")
dog2 = Dog("لونا", "بودل")

print(dog1.bark())  # ريكس يقول: واق واق!
print(dog2.bark())  # لونا يقول: واق واق!
print(dog1.name)    # ريكس
print(dog2.breed)   # بودل

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.2: المفاهيم الأساسية: Class و Object

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-004-ex13_2.py
"""

x = 5
print(type(x))        # <class 'int'>

name = "Ahmed"
print(type(name))     # <class 'str'>

my_list = [1, 2, 3]
print(type(my_list))  # <class 'list'>

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-005-__init__.py
"""

class ClassName:
    def __init__(self, parameter1, parameter2):
        self.attribute1 = parameter1
        self.attribute2 = parameter2

    def method_name(self):
        # كود يستخدم self.attribute1 وغيرها
        pass

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-006-__init__.py
"""

class Car:
    """تمثل سيارة بخصائصها الأساسية."""

    def __init__(self, brand: str, model: str, year: int):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage: int = 0  # قيمة ابتدائية افتراضية

    def drive(self, distance: int) -> None:
        """تُحرّك السيارة مسافة معينة، وتُحدّث العداد."""
        self.mileage += distance
        print(f"🚗 قطعت {self.brand} {distance} كم. المسافة الكلية: {self.mileage} كم")

    def display_info(self) -> str:
        return f"{self.year} {self.brand} {self.model} - {self.mileage} كم"

my_car = Car("تويوتا", "كورولا", 2023)
my_car.drive(150)
my_car.drive(80)
print(my_car.display_info())

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-007-my_car.py
"""

my_car = Car("تويوتا", "كورولا", 2023)
# هذا يُستدعي داخلياً:
# Car.__init__(my_car, "تويوتا", "كورولا", 2023)

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-008-__init__.py
"""

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2
print(c2.count)  # 1  -- مستقل تماماً عن c1!

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-009-ex13_3.py
"""

Counter.increment(c1)

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-010-increment.py
"""

class Counter:
    def increment():  # ❌ نسيان self!
        pass

c = Counter()
c.increment()
# 💥 TypeError: increment() takes 0 positional arguments but 1 was given

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.3: بناء أول Class: التشريح الكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-011-__init__.py
"""

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width    # attribute
        self.height = height  # attribute

    def area(self) -> float:  # method
        return self.width * self.height

    def perimeter(self) -> float:  # method
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect.width)        # 5 - الوصول لـ attribute (بدون أقواس)
print(rect.area())       # 15.0 - استدعاء method (مع أقواس)

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-012-__init__.py
"""

class Dog:
    species = "Canis familiaris"  # Class Attribute - مشتركة بين كل الكلاب

    def __init__(self, name: str):
        self.name = name  # Instance Attribute - خاصة بكل كلب

    def __str__(self) -> str:
        return f"{self.name} ({self.species})"

dog1 = Dog("ريكس")
dog2 = Dog("لونا")

print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris -- نفس القيمة للجميع

# الوصول من الفئة مباشرة أيضاً ممكن
print(Dog.species)   # Canis familiaris

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-013-ex13_4.py
"""

dog1.species = "Wolf"  # ⚠️ هذا لا يُعدّل Class Attribute!
                        # بل يُنشئ Instance Attribute جديد خاص بـ dog1 فقط!

print(dog1.species)  # Wolf (الآن Instance Attribute خاص به)
print(dog2.species)  # Canis familiaris (لم يتأثر إطلاقاً!)
print(Dog.species)   # Canis familiaris (الفئة نفسها لم تتغير)

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-014-__init__.py
"""

class Employee:
    company_name = "شركة التقنية المتقدمة"  # مشتركة للجميع
    total_employees = 0  # عداد مشترك

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1  # تعديل صحيح عبر اسم الفئة

emp1 = Employee("أحمد", 8000)
emp2 = Employee("سارة", 9500)
emp3 = Employee("عمر", 7500)

print(f"عدد الموظفين: {Employee.total_employees}")  # 3
print(f"الشركة: {emp1.company_name}")  # شركة التقنية المتقدمة

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-015-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:  # instance method
        self.balance += amount

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-016-__init__.py
"""

class Employee:
    company_name = "شركة التقنية"
    total_employees = 0

    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary
        Employee.total_employees += 1

    @classmethod
    def get_total_employees(cls) -> int:
        """تستقبل cls (الفئة نفسها) بدلاً من self (كائن محدد)."""
        return cls.total_employees

    @classmethod
    def from_string(cls, data_string: str) -> "Employee":
        """Alternative Constructor - نمط شائع جداً لإنشاء كائنات من صيغ بيانات مختلفة."""
        name, salary = data_string.split(",")
        return cls(name.strip(), float(salary.strip()))

emp1 = Employee("أحمد", 8000)
emp2 = Employee.from_string("سارة, 9500")  # إنشاء من نص!

print(emp2.name, emp2.salary)              # سارة 9500.0
print(Employee.get_total_employees())      # 2

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-017-is_prime.py
"""

class MathHelper:
    @staticmethod
    def is_prime(n: int) -> bool:
        """دالة مساعدة لا تحتاج الوصول لأي بيانات خاصة بالفئة أو الكائن."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# تُستدعى عبر الفئة مباشرة، بدون إنشاء كائن!
print(MathHelper.is_prime(17))  # True
print(MathHelper.is_prime(18))  # False

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-018-__init__.py
"""

class Pizza:
    def __init__(self, toppings: list[str]):
        self.toppings = toppings

    def instance_method(self):
        return f"هذه بيتزا بإضافات: {self.toppings}"

    @classmethod
    def margherita(cls) -> "Pizza":
        return cls(["جبنة", "طماطم"])  # Alternative constructor

    @staticmethod
    def calculate_cooking_time(size_cm: int) -> int:
        return size_cm // 2  # منطق عام لا يحتاج بيانات بيتزا معينة

p1 = Pizza.margherita()
print(p1.toppings)  # ['جبنة', 'طماطم']
print(Pizza.calculate_cooking_time(30))  # 15

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-019-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # <__main__.Point object at 0x7f8a1c0b3d90>  -- غير مفيد!

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-020-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """يُستدعى عند print() أو str() - للعرض البشري المقروء."""
        return f"نقطة({self.x}, {self.y})"

    def __repr__(self) -> str:
        """يُستدعى في REPL أو داخل قوائم - للعرض التقني الدقيق."""
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 4)
print(p)       # نقطة(3, 4)  -- يستخدم __str__
print([p])     # [Point(x=3, y=4)]  -- القوائم تستخدم __repr__ للعناصر
print(repr(p)) # Point(x=3, y=4)

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-021-__init__.py
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(5, 6)

print(p1 == p2)  # True  -- بدون __eq__ كانت ستكون False (مقارنة مراجع الذاكرة)!
print(p1 == p3)  # False

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-022-__init__.py
"""

class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(len(playlist))  # 3  -- بدون __len__ ستحصل على TypeError

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.6: Magic Methods (Dunder Methods): تخصيص سلوك الكائن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-023-__init__.py
"""

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # يستدعي v1.__add__(v2) تلقائياً!
print(v3)  # Vector(4, 6)

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-024-__init__.py
"""

class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner           # Public - وصول حر من أي مكان
        self._balance = balance      # Protected (اصطلاحاً) - "استخدم بحذر"
        self.__pin = "1234"          # Private (اصطلاحاً) - "لا تلمس من خارج الفئة"

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-025-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance  # "تلميح" أنه داخلي، لكن لا شيء يمنع الوصول فعلياً

account = BankAccount(1000)
print(account._balance)  # 1000 -- يعمل! Python لا تمنعك، لكنها تُعتبر ممارسة سيئة

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-026-__init__.py
"""

class BankAccount:
    def __init__(self, pin: str):
        self.__pin = pin

    def verify_pin(self, attempt: str) -> bool:
        return self.__pin == attempt

account = BankAccount("1234")
print(account.verify_pin("1234"))  # True

# print(account.__pin)
# 💥 AttributeError: 'BankAccount' object has no attribute '__pin'

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-027-ex13_7.py
"""

print(account._BankAccount__pin)  # 1234  -- يعمل تقنياً، لكن لا تفعل هذا أبداً!

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.7: خصوصية البيانات: Public, Protected, Private (بأسلوب Python)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-028-__init__.py
"""

class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Getter - تُستدعى عند قراءة account.balance"""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Setter - تُستدعى عند account.balance = value"""
        if value < 0:
            raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
        self._balance = value

account = BankAccount(1000)
print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

account.balance = 1500   # يستدعي الـ setter تلقائياً
print(account.balance)   # 1500

# account.balance = -500
# 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-029-__init__.py
"""

class Book:
    """يمثل كتاباً واحداً في المكتبة."""

    total_books = 0  # Class Attribute - عداد لكل الكتب المُنشأة

    def __init__(self, title: str, author: str, pages: int, is_read: bool = False):
        self.title = title
        self.author = author
        self.pages = pages
        self._is_read = is_read
        Book.total_books += 1

    @property
    def is_read(self) -> bool:
        return self._is_read

    def mark_as_read(self) -> None:
        self._is_read = True
        print(f"✅ '{self.title}' تم وضع علامة 'مقروء' عليه.")

    def __str__(self) -> str:
        status = "📖 مقروء" if self._is_read else "📕 غير مقروء"
        return f"{self.title} - {self.author} ({self.pages} صفحة) [{status}]"

    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

    @classmethod
    def from_string(cls, data: str) -> "Book":
        """ينشئ كتاباً من نص بصيغة: 'العنوان|المؤلف|الصفحات'"""
        title, author, pages = data.split("|")
        return cls(title.strip(), author.strip(), int(pages.strip()))

    @staticmethod
    def estimate_reading_hours(pages: int, pages_per_hour: int = 30) -> float:
        """تقدير عدد الساعات اللازمة لقراءة كتاب بعدد صفحات معين."""
        return round(pages / pages_per_hour, 1)

class Library:
    """تمثل مكتبة تحتوي على مجموعة من الكتب."""

    def __init__(self, name: str):
        self.name = name
        self.books: list[Book] = []

    def add_book(self, book: Book) -> None:
        if book in self.books:
            print(f"⚠️ '{book.title}' موجود مسبقاً في المكتبة.")
            return
        self.books.append(book)
        print(f"✅ تمت إضافة '{book.title}' إلى مكتبة '{self.name}'.")

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                print(f"🗑️ تم حذف '{title}'.")
                return
        print(f"❌ لم يُعثر على '{title}'.")

    def get_unread_books(self) -> list[Book]:
        return [book for book in self.books if not book.is_read]

    def total_pages(self) -> int:
        return sum(book.pages for book in self.books)

    def find_by_author(self, author: str) -> list[Book]:
        return [book for book in self.books if book.author == author]

    def __len__(self) -> int:
        return len(self.books)

    def __str__(self) -> str:
        return f"مكتبة '{self.name}' - {len(self.books)} كتاب"

    def print_all(self) -> None:
        print(f"\n📚 {self}")
        print("-" * 50)
        for i, book in enumerate(self.books, start=1):
            print(f"{i}. {book}")
        print("-" * 50)

# --- التشغيل التجريبي ---
if __name__ == "__main__":
    my_library = Library("مكتبة أحمد")

    book1 = Book("نادي السيارات", "خالد عبدالله", 250)
    book2 = Book.from_string("الخيميائي | باولو كويلو | 180")
    book3 = Book("Clean Code", "Robert Martin", 400, is_read=True)

    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book1)  # محاولة إضافة مكررة

    my_library.print_all()

    book1.mark_as_read()

    print(f"\n📖 عدد الكتب غير المقروءة: {len(my_library.get_unread_books())}")
    print(f"📄 إجمالي الصفحات: {my_library.total_pages()}")
    print(f"📊 إجمالي الكتب في كل المكتبات: {Book.total_books}")

    hours = Book.estimate_reading_hours(book2.pages)
    print(f"⏱️ وقت قراءة متوقع لـ '{book2.title}': {hours} ساعة")

    print(f"\n{my_library}")  # يستخدم __str__
    print(f"عدد الكتب (len): {len(my_library)}")  # يستخدم __len__

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-030-sort_by_pages.py
"""

def sort_by_pages(self, reverse: bool = False) -> list[Book]:
    """ترتيب كتب المكتبة حسب عدد الصفحات."""
    return sorted(self.books, key=lambda book: book.pages, reverse=reverse)

# إضافة هذا كـ method داخل Library، ثم:
sorted_books = my_library.sort_by_pages()
for book in sorted_books:
    print(f"{book.title}: {book.pages} صفحة")

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-031-__init__.py
"""

class Person:
    species = "Homo sapiens"

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self) -> None:
        print(f"مرحباً، اسمي {self.name} وعمري {self.age} سنة.")

p1 = Person("خالد", 30)
p2 = Person("منى", 25)

p1.introduce()  # مرحباً، اسمي خالد وعمري 30 سنة.
p2.introduce()  # مرحباً، اسمي منى وعمري 25 سنة.

print(p1.species)  # Homo sapiens
print(p2.species)  # Homo sapiens

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-032-__init__.py
"""

# تمرين 3
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"مستطيل {self.width}x{self.height}, المساحة={self.area()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rectangle):
            return False
        return self.area() == other.area()

r1 = Rectangle(4, 5)
r2 = Rectangle(2, 10)
print(r1)              # مستطيل 4x5, المساحة=20
print(r1 == r2)         # True -- نفس المساحة رغم اختلاف الأبعاد!

# تمرين 4
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5/9

print(TemperatureConverter.celsius_to_fahrenheit(100))  # 212.0
print(TemperatureConverter.fahrenheit_to_celsius(32))   # 0.0

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-033-__init__.py
"""

class ShoppingCart:
    def __init__(self):
        self._items: list[dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self._items.append({"name": name, "price": price, "quantity": quantity})
        print(f"✅ أُضيف {quantity}x {name} إلى السلة.")

    def total_price(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self._items)

    @property
    def item_count(self) -> int:
        """إجمالي عدد القطع (مجموع الكميات)."""
        return sum(item["quantity"] for item in self._items)

    def __len__(self) -> int:
        """عدد المنتجات المختلفة في السلة."""
        return len(self._items)

    def __str__(self) -> str:
        lines = [f"🛒 سلة التسوق ({len(self)} منتج مختلف):"]
        for item in self._items:
            subtotal = item["price"] * item["quantity"]
            lines.append(f"  - {item['name']}: {item['quantity']} × {item['price']} = {subtotal}")
        lines.append(f"الإجمالي: {self.total_price()} ر.س")
        return "\n".join(lines)

cart = ShoppingCart()
cart.add_item("لابتوب", 3500, 1)
cart.add_item("ماوس", 50, 2)
cart.add_item("كيبورد", 150, 1)

print(cart)
print(f"\nعدد القطع الكلي: {cart.item_count}")  # 4
print(f"عدد المنتجات المختلفة: {len(cart)}")    # 3

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.11: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-034-__init__.py
"""

# تعريف فئة أساسية
class ClassName:
    class_attribute = "مشترك بين الجميع"  # Class Attribute

    def __init__(self, param1, param2):
        self.attr1 = param1    # Instance Attribute
        self.attr2 = param2

    def instance_method(self):          # self = الكائن نفسه
        return self.attr1

    @classmethod
    def class_method(cls):              # cls = الفئة نفسها
        return cls.class_attribute

    @staticmethod
    def static_method():                # لا self ولا cls
        return "دالة مستقلة منطقياً"

# إنشاء كائن
obj = ClassName("value1", "value2")

# Magic Methods الأساسية
class MyClass:
    def __init__(self, x): self.x = x
    def __str__(self): return f"قابل للقراءة: {self.x}"
    def __repr__(self): return f"MyClass({self.x})"
    def __eq__(self, other): return self.x == other.x
    def __len__(self): return self.x
    def __add__(self, other): return MyClass(self.x + other.x)

# خصوصية البيانات (اصطلاحية)
self.public_attr      # وصول حر
self._protected_attr  # "تحذير" - داخلي، لا تعتمد عليه من الخارج
self.__private_attr   # Name Mangling → _ClassName__private_attr

# Property - getter/setter أنيق
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("لا يمكن أن يكون سالباً")
        self._balance = value

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
class Counter:
    def increment():  # ❌ نسيان self!
        pass

c = Counter()
c.increment()
# 💥 TypeError: increment() takes 0 positional arguments but 1 was given

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, pin: str):
        self.__pin = pin

    def verify_pin(self, attempt: str) -> bool:
        return self.__pin == attempt

account = BankAccount("1234")
print(account.verify_pin("1234"))  # True

# print(account.__pin)
# 💥 AttributeError: 'BankAccount' object has no attribute '__pin'

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    @property
    def balance(self) -> float:
        """Getter - تُستدعى عند قراءة account.balance"""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Setter - تُستدعى عند account.balance = value"""
        if value < 0:
            raise ValueError("الرصيد لا يمكن أن يكون سالباً!")
        self._balance = value

account = BankAccount(1000)
print(account.balance)   # 1000 -- يبدو كخاصية عادية، لكنه method بالخلف!

account.balance = 1500   # يستدعي الـ setter تلقائياً
print(account.balance)   # 1500

# account.balance = -500
# 💥 ValueError: الرصيد لا يمكن أن يكون سالباً!

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
class Book:
    """يمثل كتاباً واحداً في المكتبة."""

    total_books = 0  # Class Attribute - عداد لكل الكتب المُنشأة

    def __init__(self, title: str, author: str, pages: int, is_read: bool = False):
        self.title = title
        self.author = author
        self.pages = pages
        self._is_read = is_read
        Book.total_books += 1

    @property
    def is_read(self) -> bool:
        return self._is_read

    def mark_as_read(self) -> None:
        self._is_read = True
        print(f"✅ '{self.title}' تم وضع علامة 'مقروء' عليه.")

    def __str__(self) -> str:
        status = "📖 مقروء" if self._is_read else "📕 غير مقروء"
        return f"{self.title} - {self.author} ({self.pages} صفحة) [{status}]"

    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

    @classmethod
    def from_string(cls, data: str) -> "Book":
        """ينشئ كتاباً من نص بصيغة: 'العنوان|المؤلف|الصفحات'"""
        title, author, pages = data.split("|")
        return cls(title.strip(), author.strip(), int(pages.strip()))

    @staticmethod
    def estimate_reading_hours(pages: int, pages_per_hour: int = 30) -> float:
        """تقدير عدد الساعات اللازمة لقراءة كتاب بعدد صفحات معين."""
        return round(pages / pages_per_hour, 1)

class Library:
    """تمثل مكتبة تحتوي على مجموعة من الكتب."""

    def __init__(self, name: str):
        self.name = name
        self.books: list[Book] = []

    def add_book(self, book: Book) -> None:
        if book in self.books:
            print(f"⚠️ '{book.title}' موجود مسبقاً في المكتبة.")
            return
        self.books.append(book)
        print(f"✅ تمت إضافة '{book.title}' إلى مكتبة '{self.name}'.")

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                print(f"🗑️ تم حذف '{title}'.")
                return
        print(f"❌ لم يُعثر على '{title}'.")

    def get_unread_books(self) -> list[Book]:
        return [book for book in self.books if not book.is_read]

    def total_pages(self) -> int:
        return sum(book.pages for book in self.books)

    def find_by_author(self, author: str) -> list[Book]:
        return [book for book in self.books if book.author == author]

    def __len__(self) -> int:
        return len(self.books)

    def __str__(self) -> str:
        return f"مكتبة '{self.name}' - {len(self.books)} كتاب"

    def print_all(self) -> None:
        print(f"\n📚 {self}")
        print("-" * 50)
        for i, book in enumerate(self.books, start=1):
            print(f"{i}. {book}")
        print("-" * 50)

# --- التشغيل التجريبي ---
if __name__ == "__main__":
    my_library = Library("مكتبة أحمد")

    book1 = Book("نادي السيارات", "خالد عبدالله", 250)
    book2 = Book.from_string("الخيميائي | باولو كويلو | 180")
    book3 = Book("Clean Code", "Robert Martin", 400, is_read=True)

    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book1)  # محاولة إضافة مكررة

    my_library.print_all()

    book1.mark_as_read()

    print(f"\n📖 عدد الكتب غير المقروءة: {len(my_library.get_unread_books())}")
    print(f"📄 إجمالي الصفحات: {my_library.total_pages()}")
    print(f"📊 إجمالي الكتب في كل المكتبات: {Book.total_books}")

    hours = Book.estimate_reading_hours(book2.pages)
    print(f"⏱️ وقت قراءة متوقع لـ '{book2.title}': {hours} ساعة")

    print(f"\n{my_library}")  # يستخدم __str__
    print(f"عدد الكتب (len): {len(my_library)}")  # يستخدم __len__

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
def sort_by_pages(self, reverse: bool = False) -> list[Book]:
    """ترتيب كتب المكتبة حسب عدد الصفحات."""
    return sorted(self.books, key=lambda book: book.pages, reverse=reverse)

# إضافة هذا كـ method داخل Library، ثم:
sorted_books = my_library.sort_by_pages()
for book in sorted_books:
    print(f"{book.title}: {book.pages} صفحة")

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-13-chapter-13.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
# تعريف فئة أساسية
class ClassName:
    class_attribute = "مشترك بين الجميع"  # Class Attribute

    def __init__(self, param1, param2):
        self.attr1 = param1    # Instance Attribute
        self.attr2 = param2

    def instance_method(self):          # self = الكائن نفسه
        return self.attr1

    @classmethod
    def class_method(cls):              # cls = الفئة نفسها
        return cls.class_attribute

    @staticmethod
    def static_method():                # لا self ولا cls
        return "دالة مستقلة منطقياً"

# إنشاء كائن
obj = ClassName("value1", "value2")

# Magic Methods الأساسية
class MyClass:
    def __init__(self, x): self.x = x
    def __str__(self): return f"قابل للقراءة: {self.x}"
    def __repr__(self): return f"MyClass({self.x})"
    def __eq__(self, other): return self.x == other.x
    def __len__(self): return self.x
    def __add__(self, other): return MyClass(self.x + other.x)

# خصوصية البيانات (اصطلاحية)
self.public_attr      # وصول حر
self._protected_attr  # "تحذير" - داخلي، لا تعتمد عليه من الخارج
self.__private_attr   # Name Mangling → _ClassName__private_attr

# Property - getter/setter أنيق
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("لا يمكن أن يكون سالباً")
        self._balance = value
